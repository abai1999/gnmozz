#!/usr/bin/env python3
"""Audit privileged basin relabels produced by the C2C v2 frame contract path."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from prismatic.robot.coarse2contact_v2.frame_yaw_estimator import (
    frame_yaw_feature_vector,
    load_frame_yaw_checkpoint,
    resolve_yaw_observable_threshold,
)
from prismatic.robot.coarse2contact_v2.takeover_contract import (
    FrameResidual,
    ObservabilityDecision,
    TakeoverThresholds,
    classify_yaw_observability,
    explain_yaw_observability,
    decide_takeover_tier,
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _row_key(row: Mapping[str, Any]) -> tuple[int, int]:
    return (int(row.get("episode_idx", -1)), int(row.get("step_idx", row.get("step", -1))))


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _corr(a: Iterable[float], b: Iterable[float]) -> float:
    aa = np.asarray(list(a), dtype=np.float32)
    bb = np.asarray(list(b), dtype=np.float32)
    mask = np.isfinite(aa) & np.isfinite(bb)
    if np.count_nonzero(mask) < 2:
        return 0.0
    aa = aa[mask]
    bb = bb[mask]
    if np.std(aa) <= 1e-9 or np.std(bb) <= 1e-9:
        return 0.0
    return float(np.corrcoef(aa, bb)[0, 1])


def _wilson_lower_bound(successes: int, n: int, *, z: float = 1.96) -> float:
    if n <= 0:
        return 0.0
    phat = float(successes) / float(n)
    denom = 1.0 + (z * z) / float(n)
    center = phat + (z * z) / (2.0 * float(n))
    margin = z * ((phat * (1.0 - phat) + (z * z) / (4.0 * float(n))) / float(n)) ** 0.5
    return float(max((center - margin) / denom, 0.0))


def _axis_key(axis: str) -> str:
    return "dyaw" if axis == "yaw" else axis


def _row_mapping(row: Mapping[str, Any], *keys: str) -> Mapping[str, Any]:
    for key in keys:
        value = row.get(key)
        if isinstance(value, Mapping):
            return value
    return {}


def _axis_from_mapping(mapping: Mapping[str, Any], axis: str) -> float:
    key = _axis_key(axis)
    alt_key = axis if axis != "yaw" else "yaw"
    if key in mapping:
        return _safe_float(mapping.get(key, 0.0))
    if alt_key in mapping:
        return _safe_float(mapping.get(alt_key, 0.0))
    if axis == "x":
        return _safe_float(mapping.get("dx", 0.0))
    if axis == "y":
        return _safe_float(mapping.get("dy", 0.0))
    if axis == "z":
        return _safe_float(mapping.get("dz", 0.0))
    if axis == "yaw":
        return _safe_float(mapping.get("dyaw", mapping.get("yaw", 0.0)))
    return 0.0


def _value(row: Mapping[str, Any], axis: str, *, source: str = "privileged") -> float:
    key = _axis_key(axis)
    if source == "privileged":
        nested = _row_mapping(row, "true_basin_error_t")
        if nested:
            return _axis_from_mapping(nested, axis)
        return _safe_float(row.get(f"privileged_{key}", row.get(f"next_privileged_{key}", 0.0)))
    if source == "next_privileged":
        nested = _row_mapping(row, "true_basin_error_t_plus_1")
        if nested:
            return _axis_from_mapping(nested, axis)
        return _safe_float(row.get(f"next_privileged_{key}", float("nan")))
    if source == "action":
        nested = _row_mapping(row, "action_t")
        vec = np.asarray(
            nested.get("local_correction_local_6d", nested.get("planner_local_delta_6d", row.get("local_residual_vs_planner_local_6d", [0.0] * 6))),
            dtype=np.float32,
        ).reshape(-1)
        vec = np.pad(vec, (0, max(0, 6 - vec.size)))[:6]
        idx = 0 if axis == "x" else 1 if axis == "y" else 2 if axis == "z" else 5
        return float(vec[idx])
    if source == "proxy":
        proxy = row.get("proxy_local_geometry_error", {}) or {}
        return _axis_from_mapping(proxy, axis)
    if source == "estimated":
        est = row.get("estimated_basin_error", {}) or {}
        return _axis_from_mapping(est, axis)
    if source == "planner":
        nested = _row_mapping(row, "planner_prior")
        vec = np.asarray(nested.get("local_delta_6d", row.get("planner_local_delta_6d", [0.0] * 6)), dtype=np.float32).reshape(-1)
        vec = np.pad(vec, (0, max(0, 6 - vec.size)))[:6]
        idx = 0 if axis == "x" else 1 if axis == "y" else 2 if axis == "z" else 5
        return float(vec[idx])
    raise KeyError(source)


def _sequence_monotonic_rate(rows: list[dict[str, Any]], axis: str) -> float:
    if len(rows) < 2:
        return 0.0
    by_episode: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_episode[str(row.get("episode_idx", -1))].append(row)
    episode_rates: list[float] = []
    for ep_rows in by_episode.values():
        if len(ep_rows) < 2:
            continue
        vals = [abs(_value(r, axis, source="privileged")) for r in sorted(ep_rows, key=lambda r: int(r.get("step_idx", -1)))]
        if len(vals) < 2:
            continue
        episode_rates.append(float(np.mean([1.0 if vals[i] <= vals[i - 1] + 1e-9 else 0.0 for i in range(1, len(vals))])))
    return float(np.mean(episode_rates)) if episode_rates else 0.0


def _two_step_monotonic_prefix_rate(rows: list[dict[str, Any]], axis: str) -> float:
    if len(rows) < 3:
        return 0.0
    by_episode: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_episode[str(row.get("episode_idx", -1))].append(row)
    episode_rates: list[float] = []
    for ep_rows in by_episode.values():
        ordered = sorted(ep_rows, key=lambda r: int(r.get("step_idx", -1)))
        vals = [abs(_value(r, axis, source="privileged")) for r in ordered]
        if len(vals) < 3:
            continue
        prefix = [
            1.0 if (vals[idx] <= vals[idx - 1] + 1e-9 and vals[idx + 1] <= vals[idx] + 1e-9) else 0.0
            for idx in range(1, len(vals) - 1)
        ]
        if prefix:
            episode_rates.append(float(np.mean(prefix)))
    return float(np.mean(episode_rates)) if episode_rates else 0.0


def _finite_residual(row: Mapping[str, Any]) -> bool:
    return FrameResidual.from_mapping(row, source="audit_row").finite


def _visual_record_from_row(row: Mapping[str, Any]) -> dict[str, Any]:
    obs = row.get("obs_t") if isinstance(row.get("obs_t"), Mapping) else {}
    return {
        "frame_confidence": float(obs.get("frame_confidence", row.get("source_frame_confidence", 0.0)) or 0.0),
        "frame_observability": float(obs.get("frame_observability", row.get("source_frame_observability", 0.0)) or 0.0),
        "frame_axis_strength": float(obs.get("frame_axis_strength", row.get("source_frame_axis_strength", 0.0)) or 0.0),
        "wide_ring_visible": bool(obs.get("wide_ring_visible", row.get("wide_ring_visible", False))),
    }


def _observability_decision(row: Mapping[str, Any]) -> ObservabilityDecision:
    visual_class = str(row.get("visual_observability_class", row.get("obs_t", {}).get("visual_observability_class", "prior_only")))
    return classify_yaw_observability(
        row,
        _visual_record_from_row(row),
        visual_observability_class=visual_class,
    )


def _yaw_explainer(row: Mapping[str, Any]) -> dict[str, Any]:
    visual_class = str(row.get("visual_observability_class", row.get("obs_t", {}).get("visual_observability_class", "prior_only")))
    if "yaw_observability_blocker_combo" in row or "yaw_observability_primary_blocker" in row:
        gate_passes = row.get("yaw_observability_gate_passes")
        if not isinstance(gate_passes, Mapping):
            gate_passes = {}
        return {
            "visual_observability_class": visual_class,
            "frame_confidence": _safe_float(row.get("yaw_observability_frame_confidence", row.get("source_frame_confidence", 0.0)), 0.0),
            "frame_observability": _safe_float(row.get("yaw_observability_frame_observability", row.get("source_frame_observability", 0.0)), 0.0),
            "frame_axis_strength": _safe_float(row.get("yaw_observability_frame_axis_strength", row.get("source_frame_axis_strength", 0.0)), 0.0),
            "wide_ring_visible": bool(row.get("yaw_observability_wide_ring_visible", row.get("wide_ring_visible", False))),
            "wrist_is_occluded": bool(row.get("yaw_observability_wrist_occluded", row.get("wrist_is_occluded", False))),
            "gate_passes": dict(gate_passes),
            "blockers": [term for term in str(row.get("yaw_observability_blocker_combo", "")).split("+") if term],
            "blocker_combo": str(row.get("yaw_observability_blocker_combo", "")),
            "primary_blocker": str(row.get("yaw_observability_primary_blocker", "observable")),
            "reason": str(row.get("yaw_observability_reason", "")),
        }
    return explain_yaw_observability(
        row,
        _visual_record_from_row(row),
        visual_observability_class=visual_class,
    )


def _takeover_decision(row: Mapping[str, Any]) -> tuple[FrameResidual, ObservabilityDecision, Any]:
    residual = FrameResidual.from_mapping(row, source="audit_row")
    observability = _observability_decision(row)
    decision = decide_takeover_tier(
        residual,
        observability,
        precision_row=bool(str(row.get("skill_type", "")) in {"precision_grasp", "precision_align"}),
        requires_yaw_observability=bool(row.get("requires_yaw_observability", row.get("frame_contract", {}).get("requires_yaw_observability", False))),
        xy_contracted=_xy_contracted(row),
        thresholds=TakeoverThresholds(),
    )
    return residual, observability, decision


def _axis_gate_policy(row: Mapping[str, Any], axis: str) -> str:
    policy = _row_mapping(row, "axis_gate_policy")
    if policy:
        return str(policy.get(axis, "abstain"))
    if str(row.get("visual_observability_class", "")) == "prior_only":
        return "abstain"
    if axis == "yaw" and not _yaw_control_observable(row):
        return "abstain"
    if axis == "z":
        return "diagnostic_only"
    return "trusted_control"


def _micro_entry_ready(row: Mapping[str, Any]) -> bool:
    if "micro_entry_ready" in row:
        return bool(row.get("micro_entry_ready", False))
    _, _, decision = _takeover_decision(row)
    return bool(decision.micro_entry_ready)


def _micro_entry_block_reason(row: Mapping[str, Any]) -> str:
    if "micro_entry_block_reason" in row:
        return str(row.get("micro_entry_block_reason", ""))
    _, _, decision = _takeover_decision(row)
    return str(decision.micro_entry_block_reason)


def _yaw_observability_class(row: Mapping[str, Any]) -> str:
    value = str(row.get("yaw_observability_class", ""))
    if value in {"observable", "ambiguous", "unobservable"}:
        return value
    return str(_observability_decision(row).yaw_observability_class)


def _yaw_entry_feasible(row: Mapping[str, Any]) -> bool:
    if "yaw_entry_feasible" in row:
        return bool(row.get("yaw_entry_feasible", False))
    residual = FrameResidual.from_mapping(row, source="audit_row")
    return bool(residual.finite and residual.yaw_abs <= TakeoverThresholds().near_yaw + 1.0e-9)


def _yaw_control_observable(row: Mapping[str, Any]) -> bool:
    if "yaw_control_observable" in row:
        return bool(row.get("yaw_control_observable", False))
    if "yaw_observable" in row:
        return bool(row.get("yaw_observable", False))
    return bool(_yaw_observability_class(row) == "observable")


def _raw_yaw_control_observable(row: Mapping[str, Any]) -> bool:
    if "yaw_control_observable_raw" in row:
        return bool(row.get("yaw_control_observable_raw", False))
    if "yaw_control_observable" in row:
        return bool(row.get("yaw_control_observable", False))
    if "yaw_observable" in row:
        return bool(row.get("yaw_observable", False))
    return bool(_yaw_observability_class(row) == "observable")


def _yaw_blocker_combo(row: Mapping[str, Any]) -> str:
    explainer = _yaw_explainer(row)
    combo = str(explainer.get("blocker_combo", ""))
    if combo:
        return combo
    if _yaw_observability_class(row) == "observable":
        return ""
    primary = str(explainer.get("primary_blocker", "")).strip()
    return primary if primary and primary != "observable" else "low_frame_evidence"


def _yaw_blocker_terms(row: Mapping[str, Any]) -> tuple[str, ...]:
    combo = _yaw_blocker_combo(row)
    if not combo:
        return ()
    return tuple(term for term in combo.split("+") if term)


def _yaw_alignment_counts(rows: list[dict[str, Any]]) -> dict[str, Any]:
    yaw_counts = Counter(_yaw_observability_class(r) for r in rows)
    visual_counts = Counter(str(r.get("visual_observability_class", "")) for r in rows)
    blocked_rows = [r for r in rows if not _yaw_control_observable(r)]
    visual_observable_rows = [r for r in rows if str(r.get("visual_observability_class", "")) == "visual_observable"]
    visual_observable_blocked = [r for r in visual_observable_rows if not _yaw_control_observable(r)]
    visual_observable_xy_contracted_blocked = [r for r in visual_observable_blocked if _xy_contracted(r)]
    entry_feasible_rows = [r for r in rows if _yaw_entry_feasible(r)]
    control_observable_rows = [r for r in rows if _yaw_control_observable(r)]
    entry_feasible_control_blocked = [r for r in entry_feasible_rows if not _yaw_control_observable(r)]
    visual_xy_entry_feasible_control_blocked = [
        r
        for r in visual_observable_rows
        if _xy_contracted(r) and _yaw_entry_feasible(r) and not _yaw_control_observable(r)
    ]

    blocker_combos = Counter(_yaw_blocker_combo(r) for r in blocked_rows if _yaw_blocker_combo(r))
    blocker_terms = Counter(term for r in blocked_rows for term in _yaw_blocker_terms(r))
    primary_blockers = Counter(str(_yaw_explainer(r).get("primary_blocker", "")) for r in blocked_rows)
    visual_observable_blocker_combos = Counter(_yaw_blocker_combo(r) for r in visual_observable_blocked if _yaw_blocker_combo(r))
    visual_observable_xy_contracted_blocker_combos = Counter(_yaw_blocker_combo(r) for r in visual_observable_xy_contracted_blocked if _yaw_blocker_combo(r))
    visual_observable_xy_contracted_primary_blockers = Counter(str(_yaw_explainer(r).get("primary_blocker", "")) for r in visual_observable_xy_contracted_blocked)

    crosstab = []
    for visual_class, subset in sorted(_group_rows(rows, ("visual_observability_class",)).items()):
        yaw_subset_counts = Counter(_yaw_observability_class(r) for r in subset)
        blocked_subset = [r for r in subset if not _yaw_control_observable(r)]
        crosstab.append(
            {
                "visual_observability_class": visual_class[0],
                "num_rows": len(subset),
                "yaw_observability_counts": dict(yaw_subset_counts),
                "yaw_entry_feasible_rows": int(sum(_yaw_entry_feasible(r) for r in subset)),
                "yaw_control_observable_rows": int(sum(_yaw_control_observable(r) for r in subset)),
                "yaw_entry_feasible_control_blocked_rows": int(
                    sum(_yaw_entry_feasible(r) and not _yaw_control_observable(r) for r in subset)
                ),
                "yaw_blocked_rows": int(len(blocked_subset)),
                "yaw_blocked_rate": float(len(blocked_subset) / len(subset)) if subset else 0.0,
                "xy_contracted_rows": int(sum(_xy_contracted(r) for r in subset)),
                "xy_contracted_yaw_blocked_rows": int(sum(1 for r in blocked_subset if _xy_contracted(r))),
                "blocker_combo_counts": dict(Counter(_yaw_blocker_combo(r) for r in blocked_subset if _yaw_blocker_combo(r))),
                "primary_blocker_counts": dict(Counter(str(_yaw_explainer(r).get("primary_blocker", "")) for r in blocked_subset)),
            }
        )

    return {
        "yaw_observability_counts": dict(yaw_counts),
        "visual_observability_counts": dict(visual_counts),
        "yaw_entry_feasible_rows": int(len(entry_feasible_rows)),
        "yaw_entry_blocked_rows": int(max(0, len(rows) - len(entry_feasible_rows))),
        "yaw_control_observable_rows": int(len(control_observable_rows)),
        "yaw_control_blocked_rows": int(max(0, len(rows) - len(control_observable_rows))),
        "yaw_entry_feasible_control_blocked_rows": int(len(entry_feasible_control_blocked)),
        "visual_observable_xy_contracted_yaw_entry_feasible_control_blocked_rows": int(len(visual_xy_entry_feasible_control_blocked)),
        "yaw_entry_control_overlap": {
            "entry_true_control_true": int(sum(_yaw_entry_feasible(r) and _yaw_control_observable(r) for r in rows)),
            "entry_true_control_false": int(sum(_yaw_entry_feasible(r) and not _yaw_control_observable(r) for r in rows)),
            "entry_false_control_true": int(sum((not _yaw_entry_feasible(r)) and _yaw_control_observable(r) for r in rows)),
            "entry_false_control_false": int(sum((not _yaw_entry_feasible(r)) and (not _yaw_control_observable(r)) for r in rows)),
        },
        "yaw_observable_rows": int(len(control_observable_rows)),
        "yaw_blocked_rows": int(len(blocked_rows)),
        "yaw_blocked_rate": float(len(blocked_rows) / len(rows)) if rows else 0.0,
        "visual_observable_rows": int(len(visual_observable_rows)),
        "visual_observable_yaw_blocked_rows": int(len(visual_observable_blocked)),
        "visual_observable_yaw_blocked_rate": float(len(visual_observable_blocked) / len(visual_observable_rows)) if visual_observable_rows else 0.0,
        "visual_observable_xy_contracted_rows": int(sum(_xy_contracted(r) for r in visual_observable_rows)),
        "visual_observable_xy_contracted_yaw_blocked_rows": int(len(visual_observable_xy_contracted_blocked)),
        "visual_observable_xy_contracted_yaw_blocked_rate": float(len(visual_observable_xy_contracted_blocked) / len(visual_observable_rows)) if visual_observable_rows else 0.0,
        "blocker_combo_counts": dict(blocker_combos),
        "blocker_term_counts": dict(blocker_terms),
        "primary_blocker_counts": dict(primary_blockers),
        "visual_observable_blocker_combo_counts": dict(visual_observable_blocker_combos),
        "visual_observable_xy_contracted_blocker_combo_counts": dict(visual_observable_xy_contracted_blocker_combos),
        "visual_observable_xy_contracted_primary_blocker_counts": dict(visual_observable_xy_contracted_primary_blockers),
        "visual_yaw_crosstab": crosstab,
    }


def _takeover_tier(row: Mapping[str, Any]) -> str:
    value = str(row.get("takeover_tier", ""))
    if value:
        return value
    return str(_takeover_decision(row)[2].takeover_tier)


def _apply_calibrated_yaw_observability(
    rows: list[dict[str, Any]],
    *,
    checkpoint: Path,
    yaw_observable_threshold: float | None = None,
) -> dict[str, Any]:
    model, metadata = load_frame_yaw_checkpoint(checkpoint, map_location="cpu")
    resolved_threshold = float(yaw_observable_threshold) if yaw_observable_threshold is not None else resolve_yaw_observable_threshold(metadata, default=0.5)
    if not rows:
        return {
            "checkpoint": str(checkpoint.resolve()),
            "threshold": float(resolved_threshold),
            "threshold_source": "cli_override" if yaw_observable_threshold is not None else "checkpoint_metadata",
            "rows": 0,
            "calibrated_positive_rows": 0,
            "raw_positive_rows": 0,
            "mean_probability": 0.0,
        }
    features = np.stack([frame_yaw_feature_vector(r) for r in rows]).astype(np.float32)
    with torch.no_grad():
        out = model(torch.as_tensor(features, dtype=torch.float32))
    probs = out["yaw_observable_probability"].detach().cpu().numpy().astype(np.float32)
    raw_positive_rows = int(sum(_raw_yaw_control_observable(r) for r in rows))
    calibrated_positive_rows = int(np.count_nonzero(probs >= float(resolved_threshold)))
    for row, prob in zip(rows, probs):
        row["yaw_control_observable_raw"] = bool(_raw_yaw_control_observable(row))
        row["yaw_control_observable_probability"] = float(prob)
        row["yaw_control_observable_threshold"] = float(resolved_threshold)
        row["yaw_control_observable_source"] = "calibrated_checkpoint"
        row["yaw_control_observable"] = bool(float(prob) >= float(resolved_threshold))
        row["yaw_observable"] = bool(row["yaw_control_observable"])
    return {
        "checkpoint": str(checkpoint.resolve()),
        "threshold": float(resolved_threshold),
        "threshold_source": "cli_override" if yaw_observable_threshold is not None else "checkpoint_metadata",
        "rows": int(len(rows)),
        "calibrated_positive_rows": int(calibrated_positive_rows),
        "raw_positive_rows": int(raw_positive_rows),
        "delta_positive_rows": int(calibrated_positive_rows - raw_positive_rows),
        "mean_probability": float(np.mean(probs)) if probs.size else 0.0,
        "predicted_positive_rate": float(np.mean(probs >= float(resolved_threshold))) if probs.size else 0.0,
        "probability_p05_rate": float(np.mean(probs >= 0.5)) if probs.size else 0.0,
    }


def _apply_alias_drift_yaw_observability(
    rows: list[dict[str, Any]],
    *,
    alias_drift_rows_jsonl: Path,
    apply_splits: set[str] | None = None,
) -> dict[str, Any]:
    alias_rows = _read_jsonl(alias_drift_rows_jsonl)
    if apply_splits:
        alias_rows = [row for row in alias_rows if str(row.get("split", "")) in apply_splits]
    by_key: dict[tuple[int, int], dict[str, Any]] = {}
    for alias_row in alias_rows:
        key = _row_key(alias_row)
        existing = by_key.get(key)
        if existing is None or str(existing.get("split", "")) != "holdout" and str(alias_row.get("split", "")) == "holdout":
            by_key[key] = alias_row

    matched = 0
    accepted = 0
    rejected = 0
    raw_positive_rows = int(sum(_raw_yaw_control_observable(r) for r in rows))
    decision_counts: Counter[str] = Counter()
    truth_counts: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    for row in rows:
        alias_row = by_key.get(_row_key(row))
        if alias_row is None:
            row["yaw_alias_drift_decision"] = "not_scored"
            decision_counts["not_scored"] += 1
            continue

        matched += 1
        prob = _safe_float(alias_row.get("predicted_stable_alias_probability", 0.0), 0.0)
        is_accepted = bool(alias_row.get("accepted_by_stage1", alias_row.get("predicted_stable_alias", False)))
        truth_role = str(alias_row.get("acceptance_role", ""))
        split = str(alias_row.get("split", ""))
        decision = "stable_alias_control" if is_accepted else "frame_drift_abstain"

        row["yaw_control_observable_raw"] = bool(_raw_yaw_control_observable(row))
        row["yaw_control_observable_probability"] = float(prob)
        row["yaw_control_observable_source"] = "alias_drift_two_stage"
        row["yaw_alias_drift_decision"] = decision
        row["yaw_alias_drift_truth_role"] = truth_role
        row["yaw_alias_drift_split"] = split
        row["yaw_alias_drift_raw_proxy_yaw"] = _safe_float(alias_row.get("raw_proxy_yaw", float("nan")), float("nan"))
        row["yaw_alias_drift_symmetry_aware_proxy_yaw"] = _safe_float(alias_row.get("symmetry_aware_proxy_yaw", float("nan")), float("nan"))
        row["yaw_alias_drift_regressed_dyaw"] = _safe_float(alias_row.get("regressed_dyaw", float("nan")), float("nan"))
        row["yaw_control_observable"] = bool(is_accepted)
        row["yaw_observable"] = bool(is_accepted)
        if is_accepted:
            row["yaw_observability_class"] = "observable"

        accepted += int(is_accepted)
        rejected += int(not is_accepted)
        decision_counts[decision] += 1
        truth_counts[truth_role] += 1
        split_counts[split] += 1

    return {
        "rows_jsonl": str(alias_drift_rows_jsonl.resolve()),
        "apply_splits": sorted(apply_splits) if apply_splits else ["all"],
        "input_scored_rows": int(len(alias_rows)),
        "rows": int(len(rows)),
        "matched_rows": int(matched),
        "matched_rate": float(matched / len(rows)) if rows else 0.0,
        "raw_positive_rows": int(raw_positive_rows),
        "alias_drift_positive_rows": int(accepted),
        "alias_drift_rejected_rows": int(rejected),
        "delta_positive_rows": int(accepted - raw_positive_rows),
        "decision_counts": dict(decision_counts),
        "truth_role_counts": dict(truth_counts),
        "split_counts": dict(split_counts),
    }


def _xy_contracted(row: Mapping[str, Any]) -> bool:
    xy = _safe_float(row.get("xy_error", FrameResidual.from_mapping(row, source="audit_row").xy_error), float("nan"))
    nxt = _safe_float(row.get("next_xy_error", float("nan")), float("nan"))
    return bool(np.isfinite(xy) and np.isfinite(nxt) and nxt < xy - 1.0e-9)


def _row_overshoot(row: Mapping[str, Any]) -> bool:
    if "overshoot" in row:
        return bool(row.get("overshoot", False))
    for axis in ("x", "y", "yaw"):
        now = _value(row, axis, source="privileged")
        nxt = _value(row, axis, source="next_privileged")
        if np.isfinite(now) and np.isfinite(nxt) and np.sign(now) != np.sign(nxt) and abs(nxt) >= abs(now):
            return True
    return False


def _yaw_audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    classes = Counter(_yaw_observability_class(r) for r in rows)
    observable = [r for r in rows if _yaw_control_observable(r)]
    entry_feasible = [r for r in rows if _yaw_entry_feasible(r)]
    unobservable = [r for r in rows if _yaw_observability_class(r) == "unobservable"]
    proxy = [_value(r, "yaw", source="proxy") for r in rows]
    priv = [_value(r, "yaw", source="privileged") for r in rows]
    abstain_correct = [
        _axis_gate_policy(r, "yaw") == "abstain"
        for r in rows
        if _yaw_observability_class(r) != "observable"
    ]
    return {
        "yaw_observability_counts": dict(classes),
        "yaw_observable_rate": float(len(observable) / len(rows)) if rows else 0.0,
        "yaw_blocked_rate": float(1.0 - len(observable) / len(rows)) if rows else 0.0,
        "yaw_entry_feasible_rate": float(len(entry_feasible) / len(rows)) if rows else 0.0,
        "yaw_control_observable_rate": float(len(observable) / len(rows)) if rows else 0.0,
        "yaw_proxy_vs_privileged_error": float(np.nanmean(np.abs(np.asarray(proxy, dtype=np.float32) - np.asarray(priv, dtype=np.float32)))) if rows else 0.0,
        "yaw_proxy_priv_corr": _corr(proxy, priv),
        "yaw_abstain_correct_rate": float(np.mean(abstain_correct)) if abstain_correct else 1.0,
        "unobservable_rows": int(len(unobservable)),
    }


def _tier_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    tiers = Counter(_takeover_tier(r) for r in rows)
    contracted_total = int(sum(_xy_contracted(r) for r in rows))
    out: dict[str, Any] = {
        "takeover_tier_counts": dict(tiers),
        "coarse_pullback_candidate_rows": int(tiers.get("coarse_pullback_candidate", 0)),
        "outer_pullback_candidate_rows": int(tiers.get("outer_pullback_candidate", 0)),
        "near_basin_shell_rows": int(sum(bool(r.get("near_basin_shell", False)) for r in rows)),
        "near_basin_shell_tier_rows": int(tiers.get("near_basin_shell", 0)),
        "micro_entry_ready_rows": int(sum(_micro_entry_ready(r) for r in rows)),
        "close_ready_rows": int(sum(bool(r.get("close_ready_ready", False)) for r in rows)),
        "close_ready_tier_rows": int(tiers.get("close_ready", 0)),
        "xy_contracted_count": int(contracted_total),
        "xy_contraction_lower_ci": float(_wilson_lower_bound(contracted_total, len(rows))),
    }
    by_tier: list[dict[str, Any]] = []
    for key, subset in sorted(_group_rows(rows, ("takeover_tier",)).items()):
        tier_rows = subset
        if key[0] == "":
            tier_rows = [r for r in rows if _takeover_tier(r) == "outside_takeover"]
        contracted_count = int(sum(_xy_contracted(r) for r in tier_rows))
        by_tier.append(
            {
                "takeover_tier": key[0] or "outside_takeover",
                "num_rows": int(len(tier_rows)),
                "xy_contracted_count": int(contracted_count),
                "xy_contraction_rate": float(np.mean([_xy_contracted(r) for r in tier_rows])) if tier_rows else 0.0,
                "xy_contraction_lower_ci": float(_wilson_lower_bound(contracted_count, len(tier_rows))),
                "overshoot_rate": float(np.mean([_row_overshoot(r) for r in tier_rows])) if tier_rows else 0.0,
                "near_grasp_rate": float(np.mean([bool(r.get("near_grasp_basin", False)) for r in tier_rows])) if tier_rows else 0.0,
                "prior_only_abstain_rate": float(np.mean([str(r.get("visual_observability_class", "")) == "prior_only" and _axis_gate_policy(r, "x") == "abstain" for r in tier_rows])) if tier_rows else 0.0,
                "yaw_entry_feasible_rate": float(np.mean([_yaw_entry_feasible(r) for r in tier_rows])) if tier_rows else 0.0,
                "yaw_control_observable_rate": float(np.mean([_yaw_control_observable(r) for r in tier_rows])) if tier_rows else 0.0,
                "yaw_observable_rate": float(np.mean([_yaw_control_observable(r) for r in tier_rows])) if tier_rows else 0.0,
                "yaw_blocked_count": int(sum(1 for r in tier_rows if not _yaw_control_observable(r))),
            }
        )
    out["by_takeover_tier"] = by_tier
    return out


def _axis_stats(rows: list[dict[str, Any]], axis: str) -> dict[str, Any]:
    proxy = np.asarray([_value(r, axis, source="proxy") for r in rows], dtype=np.float32)
    est = np.asarray([_value(r, axis, source="estimated") for r in rows], dtype=np.float32)
    priv = np.asarray([_value(r, axis, source="privileged") for r in rows], dtype=np.float32)
    action = np.asarray([_value(r, axis, source="action") for r in rows], dtype=np.float32)
    next_priv = np.asarray([_value(r, axis, source="next_privileged") for r in rows], dtype=np.float32)
    planner = np.asarray([_value(r, axis, source="planner") for r in rows], dtype=np.float32)

    finite = np.isfinite(proxy) & np.isfinite(priv)
    finite_est = np.isfinite(est) & np.isfinite(priv)
    finite_action = np.isfinite(action) & np.isfinite(priv)
    finite_next = np.isfinite(next_priv) & np.isfinite(priv)
    trusted_mask = np.asarray([_axis_gate_policy(r, axis) == "trusted_control" for r in rows], dtype=bool)

    sign_mask = finite & (np.abs(proxy) > 1e-6) & (np.abs(priv) > 1e-6)
    action_sign_mask = finite_action & (np.abs(action) > 1e-6) & (np.abs(priv) > 1e-6)
    sign_match = float(np.mean([np.sign(proxy[i]) == np.sign(priv[i]) for i in np.where(sign_mask)[0]])) if np.any(sign_mask) else 0.0
    action_sign_match = float(np.mean([np.sign(action[i]) == np.sign(priv[i]) for i in np.where(action_sign_mask)[0]])) if np.any(action_sign_mask) else 0.0
    contraction = float(np.mean(np.abs(next_priv[finite_next]) <= np.abs(priv[finite_next]) + 1e-9)) if np.any(finite_next) else 0.0
    overshoot = float(np.mean((np.sign(next_priv[finite_next]) != np.sign(priv[finite_next])) & (np.abs(next_priv[finite_next]) >= np.abs(priv[finite_next])))) if np.any(finite_next) else 0.0
    trusted_proxy = proxy[trusted_mask]
    trusted_priv = priv[trusted_mask]
    trusted_next = next_priv[trusted_mask]
    trusted_finite = np.isfinite(trusted_proxy) & np.isfinite(trusted_priv)
    trusted_next_finite = np.isfinite(trusted_next) & np.isfinite(trusted_priv)
    trusted_sign_mask = trusted_finite & (np.abs(trusted_proxy) > 1e-6) & (np.abs(trusted_priv) > 1e-6)
    trusted_sign_match = float(np.mean([np.sign(trusted_proxy[i]) == np.sign(trusted_priv[i]) for i in np.where(trusted_sign_mask)[0]])) if np.any(trusted_sign_mask) else 0.0
    trusted_contraction = float(np.mean(np.abs(trusted_next[trusted_next_finite]) <= np.abs(trusted_priv[trusted_next_finite]) + 1e-9)) if np.any(trusted_next_finite) else 0.0

    trusted_policy = "trusted_control"
    if not (sign_match >= 0.70 and contraction >= 0.70 and abs(_corr(proxy, priv)) >= 0.25):
        trusted_policy = "diagnostic_only" if sign_match >= 0.40 and contraction >= 0.40 else "abstain"

    return {
        "num_rows": int(len(rows)),
        "sign_match_rate": sign_match,
        "action_sign_match_rate": action_sign_match,
        "contraction_rate": contraction,
        "one_step_contraction_rate": contraction,
        "two_step_monotonic_prefix_rate": _two_step_monotonic_prefix_rate(rows, axis),
        "overshoot_rate": overshoot,
        "monotonic_prefix_rate": _sequence_monotonic_rate(rows, axis),
        "proxy_priv_corr": _corr(proxy, priv),
        "estimated_priv_corr": _corr(est, priv),
        "action_priv_corr": _corr(action, priv),
        "planner_priv_corr": _corr(planner, priv),
        "recommended_policy": trusted_policy,
        "trusted_rows": int(np.count_nonzero(trusted_mask)),
        "trusted_sign_match_rate": trusted_sign_match,
        "trusted_contraction_rate": trusted_contraction,
        "trusted_two_step_monotonic_prefix_rate": _two_step_monotonic_prefix_rate([r for r, keep in zip(rows, trusted_mask) if bool(keep)], axis),
    }


def _group_rows(rows: list[dict[str, Any]], keys: tuple[str, ...]) -> dict[tuple[str, ...], list[dict[str, Any]]]:
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(str(row.get(k, "")) for k in keys)].append(row)
    return groups


def _window_protocol_key(row: Mapping[str, Any]) -> tuple[str, str, str, str]:
    window = row.get("window_protocol") if isinstance(row.get("window_protocol"), Mapping) else {}
    if window:
        return (
            str(window.get("window_mode", "")),
            str(window.get("shell_filter", "")),
            "true" if bool(window.get("queue_flushed", False)) else "false",
            str(window.get("requested_horizon", "")),
        )
    return (
        str(row.get("grasp_probe_window_mode", row.get("c2c_grasp_probe_window_mode", ""))),
        str(row.get("grasp_probe_shell_filter", row.get("c2c_grasp_probe_shell_filter", ""))),
        "true" if bool(row.get("grasp_probe_queue_flushed", False)) else "false",
        str(row.get("grasp_probe_requested_horizon", row.get("c2c_grasp_probe_horizon", ""))),
    )


def _plot_overview(report: dict[str, Any], output_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    axes = ["x", "y", "z", "yaw"]
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    axs = axs.reshape(-1)
    for ax, axis in zip(axs, axes):
        stats = report["axis_summary"][axis]
        bars = [stats["sign_match_rate"], stats["contraction_rate"], abs(stats["proxy_priv_corr"]), abs(stats["action_priv_corr"])]
        labels = ["sign", "contract", "proxy", "action"]
        ax.bar(labels, bars, color=["#4e79a7", "#59a14f", "#f28e2b", "#e15759"])
        ax.set_ylim(0.0, 1.0)
        ax.set_title(f"{axis}: {stats['recommended_policy']}")
    fig.tight_layout()
    fig.savefig(output_dir / "frame_contract_relabel_overview.png", dpi=160)
    plt.close(fig)


def audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    axis_summary = {axis: _axis_stats(rows, axis) for axis in ["x", "y", "z", "yaw"]}

    micro_entry_reasons = Counter(_micro_entry_block_reason(r) for r in rows)
    yaw_summary = _yaw_audit(rows)
    yaw_alignment = _yaw_alignment_counts(rows)
    tier_summary = _tier_summary(rows)

    by_episode = []
    for key, subset in sorted(_group_rows(rows, ("episode_idx",)).items(), key=lambda item: int(item[0][0]) if str(item[0][0]).lstrip("-").isdigit() else -1):
        by_episode.append(
            {
                "episode_idx": int(key[0]) if str(key[0]).lstrip("-").isdigit() else -1,
                "num_rows": len(subset),
                "axis_summary": {axis: _axis_stats(subset, axis) for axis in ["x", "y", "z", "yaw"]},
                "micro_entry_ready_rate": float(np.mean([_micro_entry_ready(r) for r in subset])) if subset else 0.0,
                "takeover_tier_counts": dict(Counter(_takeover_tier(r) for r in subset)),
                "yaw_observability_counts": dict(Counter(_yaw_observability_class(r) for r in subset)),
                "failure_bucket_counts": dict(Counter(str(r.get("failure_bucket", "")) for r in subset)),
            }
        )

    by_stage = []
    for key, subset in sorted(_group_rows(rows, ("stage_name",)).items()):
        subset_yaw = _yaw_alignment_counts(subset)
        by_stage.append(
            {
                "stage_name": key[0],
                "axis_summary": {axis: _axis_stats(subset, axis) for axis in ["x", "y", "z", "yaw"]},
                "micro_entry_ready_rate": float(np.mean([_micro_entry_ready(r) for r in subset])) if subset else 0.0,
                "micro_entry_block_reason_counts": dict(Counter(_micro_entry_block_reason(r) for r in subset)),
                "yaw_observability_counts": subset_yaw["yaw_observability_counts"],
                "yaw_blocked_rows": subset_yaw["yaw_blocked_rows"],
                "visual_observable_yaw_blocked_rows": subset_yaw["visual_observable_yaw_blocked_rows"],
                "visual_observable_xy_contracted_yaw_blocked_rows": subset_yaw["visual_observable_xy_contracted_yaw_blocked_rows"],
                "yaw_blocker_combo_counts": subset_yaw["blocker_combo_counts"],
                "yaw_primary_blocker_counts": subset_yaw["primary_blocker_counts"],
            }
        )

    by_skill = []
    for key, subset in sorted(_group_rows(rows, ("skill_name",)).items()):
        subset_yaw = _yaw_alignment_counts(subset)
        by_skill.append(
            {
                "skill_name": key[0],
                "axis_summary": {axis: _axis_stats(subset, axis) for axis in ["x", "y", "z", "yaw"]},
                "micro_entry_ready_rate": float(np.mean([_micro_entry_ready(r) for r in subset])) if subset else 0.0,
                "micro_entry_block_reason_counts": dict(Counter(_micro_entry_block_reason(r) for r in subset)),
                "yaw_observability_counts": subset_yaw["yaw_observability_counts"],
                "yaw_blocked_rows": subset_yaw["yaw_blocked_rows"],
                "visual_observable_yaw_blocked_rows": subset_yaw["visual_observable_yaw_blocked_rows"],
                "visual_observable_xy_contracted_yaw_blocked_rows": subset_yaw["visual_observable_xy_contracted_yaw_blocked_rows"],
                "yaw_blocker_combo_counts": subset_yaw["blocker_combo_counts"],
                "yaw_primary_blocker_counts": subset_yaw["primary_blocker_counts"],
            }
        )

    by_visual = []
    for key, subset in sorted(_group_rows(rows, ("visual_observability_class",)).items()):
        subset_yaw = _yaw_alignment_counts(subset)
        by_visual.append(
            {
                "visual_observability_class": key[0],
                "axis_summary": {axis: _axis_stats(subset, axis) for axis in ["x", "y", "z", "yaw"]},
                "micro_entry_ready_rate": float(np.mean([_micro_entry_ready(r) for r in subset])) if subset else 0.0,
                "micro_entry_block_reason_counts": dict(Counter(_micro_entry_block_reason(r) for r in subset)),
                "yaw_observability_counts": subset_yaw["yaw_observability_counts"],
                "yaw_blocked_rows": subset_yaw["yaw_blocked_rows"],
                "visual_observable_yaw_blocked_rows": subset_yaw["visual_observable_yaw_blocked_rows"],
                "visual_observable_xy_contracted_yaw_blocked_rows": subset_yaw["visual_observable_xy_contracted_yaw_blocked_rows"],
                "yaw_blocker_combo_counts": subset_yaw["blocker_combo_counts"],
                "yaw_primary_blocker_counts": subset_yaw["primary_blocker_counts"],
            }
        )

    by_bucket = []
    for key, subset in sorted(_group_rows(rows, ("failure_bucket",)).items()):
        subset_yaw = _yaw_alignment_counts(subset)
        by_bucket.append(
            {
                "failure_bucket": key[0],
                "axis_summary": {axis: _axis_stats(subset, axis) for axis in ["x", "y", "z", "yaw"]},
                "micro_entry_ready_rate": float(np.mean([_micro_entry_ready(r) for r in subset])) if subset else 0.0,
                "micro_entry_block_reason_counts": dict(Counter(_micro_entry_block_reason(r) for r in subset)),
                "yaw_observability_counts": subset_yaw["yaw_observability_counts"],
                "yaw_blocked_rows": subset_yaw["yaw_blocked_rows"],
                "visual_observable_yaw_blocked_rows": subset_yaw["visual_observable_yaw_blocked_rows"],
                "visual_observable_xy_contracted_yaw_blocked_rows": subset_yaw["visual_observable_xy_contracted_yaw_blocked_rows"],
                "yaw_blocker_combo_counts": subset_yaw["blocker_combo_counts"],
                "yaw_primary_blocker_counts": subset_yaw["primary_blocker_counts"],
            }
        )

    by_yaw_observability = []
    for key, subset in sorted(_group_rows(rows, ("yaw_observability_class",)).items()):
        cls = key[0] or "unknown"
        by_yaw_observability.append(
            {
                "yaw_observability_class": cls,
                "num_rows": len(subset),
                "axis_summary": {axis: _axis_stats(subset, axis) for axis in ["x", "y", "z", "yaw"]},
                "takeover_tier_counts": dict(Counter(_takeover_tier(r) for r in subset)),
            }
        )

    by_yaw_alias_drift_decision = []
    for key, subset in sorted(_group_rows(rows, ("yaw_alias_drift_decision",)).items()):
        decision = key[0] or "not_available"
        by_yaw_alias_drift_decision.append(
            {
                "yaw_alias_drift_decision": decision,
                "num_rows": len(subset),
                "axis_summary": {axis: _axis_stats(subset, axis) for axis in ["x", "y", "z", "yaw"]},
                "takeover_tier_counts": dict(Counter(_takeover_tier(r) for r in subset)),
                "yaw_observability_counts": dict(Counter(_yaw_observability_class(r) for r in subset)),
                "yaw_control_observable_rows": int(sum(_yaw_control_observable(r) for r in subset)),
                "yaw_entry_feasible_rows": int(sum(_yaw_entry_feasible(r) for r in subset)),
                "xy_contracted_rows": int(sum(_xy_contracted(r) for r in subset)),
                "truth_role_counts": dict(Counter(str(r.get("yaw_alias_drift_truth_role", "")) for r in subset)),
                "split_counts": dict(Counter(str(r.get("yaw_alias_drift_split", "")) for r in subset)),
            }
        )

    by_takeover_tier = []
    tier_groups: dict[tuple[str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        tier_groups[(_takeover_tier(row),)].append(row)
    for key, subset in sorted(tier_groups.items()):
        contracted_count = int(sum(_xy_contracted(r) for r in subset))
        by_takeover_tier.append(
            {
                "takeover_tier": key[0],
                "num_rows": len(subset),
                "xy_contracted_count": int(contracted_count),
                "axis_summary": {axis: _axis_stats(subset, axis) for axis in ["x", "y", "z", "yaw"]},
                "xy_contraction_rate": float(np.mean([_xy_contracted(r) for r in subset])) if subset else 0.0,
                "xy_contraction_lower_ci": float(_wilson_lower_bound(contracted_count, len(subset))),
                "overshoot_rate": float(np.mean([_row_overshoot(r) for r in subset])) if subset else 0.0,
                "yaw_observability_counts": dict(Counter(_yaw_observability_class(r) for r in subset)),
            }
        )

    by_window_protocol = []
    window_groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        window_groups[_window_protocol_key(row)].append(row)
    for key, subset in sorted(window_groups.items()):
        by_window_protocol.append(
            {
                "window_mode": key[0],
                "shell_filter": key[1] if len(key) > 1 else "",
                "queue_flushed": key[2] if len(key) > 2 else "",
                "requested_horizon": key[3] if len(key) > 3 else "",
                "num_rows": len(subset),
                "axis_summary": {axis: _axis_stats(subset, axis) for axis in ["x", "y", "z", "yaw"]},
                "micro_entry_ready_rate": float(np.mean([_micro_entry_ready(r) for r in subset])) if subset else 0.0,
                "takeover_tier_counts": dict(Counter(_takeover_tier(r) for r in subset)),
            }
        )

    overall = {
        "num_rows": len(rows),
        "near_grasp_rate": float(np.mean([bool(r.get("near_grasp_basin", False)) for r in rows])) if rows else 0.0,
        "close_ready_rate": float(np.mean([bool(r.get("close_ready_basin", False)) for r in rows])) if rows else 0.0,
        "visual_observable_rate": float(np.mean([str(r.get("visual_observability_class", "")) == "visual_observable" for r in rows])) if rows else 0.0,
        "prior_only_rate": float(np.mean([str(r.get("visual_observability_class", "")) == "prior_only" for r in rows])) if rows else 0.0,
        "micro_entry_ready_rate": float(np.mean([_micro_entry_ready(r) for r in rows])) if rows else 0.0,
        "micro_entry_block_reason_counts": dict(micro_entry_reasons),
        "schema_version_counts": dict(Counter(str(r.get("schema_version", "legacy")) for r in rows)),
        "label_valid_rate": float(np.mean([bool(r.get("label_valid", True)) for r in rows])) if rows else 0.0,
        "yaw_entry_feasible_rows": int(sum(_yaw_entry_feasible(r) for r in rows)),
        "yaw_entry_blocked_rows": int(sum(not _yaw_entry_feasible(r) for r in rows)),
        "yaw_control_observable_rows": int(sum(_yaw_control_observable(r) for r in rows)),
        "yaw_control_blocked_rows": int(sum(not _yaw_control_observable(r) for r in rows)),
        "yaw_entry_feasible_control_blocked_rows": int(sum(_yaw_entry_feasible(r) and not _yaw_control_observable(r) for r in rows)),
        "yaw_observable_rows": int(sum(_yaw_control_observable(r) for r in rows)),
        "yaw_blocked_rows": int(sum(not _yaw_control_observable(r) for r in rows)),
        "visual_observable_yaw_blocked_rows": int(yaw_alignment["visual_observable_yaw_blocked_rows"]),
        "visual_observable_xy_contracted_yaw_blocked_rows": int(yaw_alignment["visual_observable_xy_contracted_yaw_blocked_rows"]),
        "visual_observable_xy_contracted_yaw_entry_feasible_control_blocked_rows": int(
            yaw_alignment["visual_observable_xy_contracted_yaw_entry_feasible_control_blocked_rows"]
        ),
        "visual_observable_yaw_blocked_rate": float(yaw_alignment["visual_observable_yaw_blocked_rate"]),
        "visual_observable_xy_contracted_yaw_blocked_rate": float(yaw_alignment["visual_observable_xy_contracted_yaw_blocked_rate"]),
        "near_basin_shell_yaw_entry_feasible_rows": int(sum(1 for r in rows if bool(r.get("near_basin_shell", False)) and _yaw_entry_feasible(r))),
        "near_basin_shell_yaw_entry_blocked_rows": int(sum(1 for r in rows if bool(r.get("near_basin_shell", False)) and not _yaw_entry_feasible(r))),
        "near_basin_shell_yaw_control_observable_rows": int(sum(1 for r in rows if bool(r.get("near_basin_shell", False)) and _yaw_control_observable(r))),
        "near_basin_shell_yaw_control_blocked_rows": int(sum(1 for r in rows if bool(r.get("near_basin_shell", False)) and not _yaw_control_observable(r))),
        "near_basin_shell_yaw_observable_rows": int(sum(1 for r in rows if bool(r.get("near_basin_shell", False)) and _yaw_control_observable(r))),
        "near_basin_shell_yaw_blocked_rows": int(sum(1 for r in rows if bool(r.get("near_basin_shell", False)) and not _yaw_control_observable(r))),
        "near_basin_shell_yaw_observable_rate": float(
            np.mean([_yaw_control_observable(r) for r in rows if bool(r.get("near_basin_shell", False))])
        ) if any(bool(r.get("near_basin_shell", False)) for r in rows) else 0.0,
        "xy_contracted_count": int(sum(_xy_contracted(r) for r in rows)),
        "xy_contraction_lower_ci": float(_wilson_lower_bound(int(sum(_xy_contracted(r) for r in rows)), len(rows))),
        "yaw_blocker_combo_counts": dict(yaw_alignment["blocker_combo_counts"]),
        "yaw_primary_blocker_counts": dict(yaw_alignment["primary_blocker_counts"]),
        "visual_yaw_crosstab": yaw_alignment["visual_yaw_crosstab"],
        **yaw_summary,
        **{k: v for k, v in tier_summary.items() if k != "by_takeover_tier"},
    }

    return {
        "overall": overall,
        "yaw_alignment": yaw_alignment,
        "axis_summary": axis_summary,
        "by_episode": by_episode,
        "by_stage": by_stage,
        "by_skill": by_skill,
        "by_visual_observability": by_visual,
        "by_yaw_observability": by_yaw_observability,
        "by_yaw_alias_drift_decision": by_yaw_alias_drift_decision,
        "by_takeover_tier": by_takeover_tier,
        "by_failure_bucket": by_bucket,
        "by_window_protocol": by_window_protocol,
        "runtime_invariants": {
            "uses_privileged_target": False,
            "uses_privileged_runtime": False,
            "uses_privileged_label": True,
            "uses_rlbench_mask_runtime": False,
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--relabel_jsonl", type=Path, required=True)
    ap.add_argument("--yaw_observability_checkpoint", type=Path, default=None)
    ap.add_argument("--yaw_observability_threshold", type=float, default=None)
    ap.add_argument("--yaw_alias_drift_rows_jsonl", type=Path, default=None)
    ap.add_argument(
        "--yaw_alias_drift_apply_splits",
        type=str,
        default="holdout",
        help="Comma-separated baseline splits to apply to audit rows, or 'all'. Default is holdout.",
    )
    ap.add_argument(
        "--output_dir",
        type=Path,
        default=Path("runtime_artifacts/coarse2contact_v2/reports/frame_contract_relabel"),
    )
    args = ap.parse_args()

    rows = _read_jsonl(args.relabel_jsonl)
    if not rows:
        raise RuntimeError(f"No rows found in {args.relabel_jsonl}")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_rows = [dict(row) for row in rows]
    raw_report = audit(raw_rows)
    report = raw_report
    calibration_summary: dict[str, Any] | None = None
    alias_drift_summary: dict[str, Any] | None = None
    if args.yaw_observability_checkpoint is not None:
        calibrated_rows = [dict(row) for row in rows]
        calibration_summary = _apply_calibrated_yaw_observability(
            calibrated_rows,
            checkpoint=args.yaw_observability_checkpoint,
            yaw_observable_threshold=args.yaw_observability_threshold,
        )
        report = audit(calibrated_rows)
        report["calibration"] = calibration_summary
        report["raw_reference"] = {
            "source_jsonl": str(args.relabel_jsonl.resolve()),
            "overall": {
                "yaw_control_observable_rows": int(raw_report["overall"]["yaw_control_observable_rows"]),
                "yaw_control_observable_rate": float(raw_report["overall"]["yaw_control_observable_rate"]),
                "yaw_control_blocked_rows": int(raw_report["overall"]["yaw_control_blocked_rows"]),
                "yaw_entry_feasible_control_blocked_rows": int(raw_report["overall"]["yaw_entry_feasible_control_blocked_rows"]),
                "visual_observable_yaw_blocked_rows": int(raw_report["overall"]["visual_observable_yaw_blocked_rows"]),
                "visual_observable_xy_contracted_yaw_blocked_rows": int(raw_report["overall"]["visual_observable_xy_contracted_yaw_blocked_rows"]),
                "near_basin_shell_yaw_control_observable_rows": int(raw_report["overall"]["near_basin_shell_yaw_control_observable_rows"]),
                "near_basin_shell_yaw_control_blocked_rows": int(raw_report["overall"]["near_basin_shell_yaw_control_blocked_rows"]),
                "near_basin_shell_yaw_observable_rate": float(raw_report["overall"]["near_basin_shell_yaw_observable_rate"]),
                "visual_observable_yaw_blocked_rate": float(raw_report["overall"]["visual_observable_yaw_blocked_rate"]),
            },
        }
        report["comparison"] = {
            "yaw_control_observable_rows_delta": int(report["overall"]["yaw_control_observable_rows"] - raw_report["overall"]["yaw_control_observable_rows"]),
            "yaw_control_observable_rate_delta": float(report["overall"]["yaw_control_observable_rate"] - raw_report["overall"]["yaw_control_observable_rate"]),
            "yaw_entry_feasible_control_blocked_rows_delta": int(
                report["overall"]["yaw_entry_feasible_control_blocked_rows"] - raw_report["overall"]["yaw_entry_feasible_control_blocked_rows"]
            ),
            "visual_observable_yaw_blocked_rows_delta": int(
                report["overall"]["visual_observable_yaw_blocked_rows"] - raw_report["overall"]["visual_observable_yaw_blocked_rows"]
            ),
            "visual_observable_xy_contracted_yaw_blocked_rows_delta": int(
                report["overall"]["visual_observable_xy_contracted_yaw_blocked_rows"]
                - raw_report["overall"]["visual_observable_xy_contracted_yaw_blocked_rows"]
            ),
            "near_basin_shell_yaw_control_observable_rows_delta": int(
                report["overall"]["near_basin_shell_yaw_control_observable_rows"]
                - raw_report["overall"]["near_basin_shell_yaw_control_observable_rows"]
            ),
            "near_basin_shell_yaw_control_blocked_rows_delta": int(
                report["overall"]["near_basin_shell_yaw_control_blocked_rows"]
                - raw_report["overall"]["near_basin_shell_yaw_control_blocked_rows"]
            ),
            "near_basin_shell_yaw_observable_rate_delta": float(
                report["overall"]["near_basin_shell_yaw_observable_rate"] - raw_report["overall"]["near_basin_shell_yaw_observable_rate"]
            ),
        }
    elif args.yaw_alias_drift_rows_jsonl is not None:
        alias_rows = [dict(row) for row in rows]
        split_terms = [term.strip() for term in str(args.yaw_alias_drift_apply_splits).split(",") if term.strip()]
        apply_splits = None if not split_terms or split_terms == ["all"] else set(split_terms)
        alias_drift_summary = _apply_alias_drift_yaw_observability(
            alias_rows,
            alias_drift_rows_jsonl=args.yaw_alias_drift_rows_jsonl,
            apply_splits=apply_splits,
        )
        report = audit(alias_rows)
        report["alias_drift_yaw_observability"] = alias_drift_summary
        report["raw_reference"] = {
            "source_jsonl": str(args.relabel_jsonl.resolve()),
            "overall": {
                "yaw_control_observable_rows": int(raw_report["overall"]["yaw_control_observable_rows"]),
                "yaw_control_observable_rate": float(raw_report["overall"]["yaw_control_observable_rate"]),
                "yaw_control_blocked_rows": int(raw_report["overall"]["yaw_control_blocked_rows"]),
                "yaw_entry_feasible_control_blocked_rows": int(raw_report["overall"]["yaw_entry_feasible_control_blocked_rows"]),
                "near_basin_shell_yaw_control_observable_rows": int(raw_report["overall"]["near_basin_shell_yaw_control_observable_rows"]),
                "near_basin_shell_yaw_control_blocked_rows": int(raw_report["overall"]["near_basin_shell_yaw_control_blocked_rows"]),
            },
        }
        report["comparison"] = {
            "yaw_control_observable_rows_delta": int(report["overall"]["yaw_control_observable_rows"] - raw_report["overall"]["yaw_control_observable_rows"]),
            "yaw_control_observable_rate_delta": float(report["overall"]["yaw_control_observable_rate"] - raw_report["overall"]["yaw_control_observable_rate"]),
            "yaw_entry_feasible_control_blocked_rows_delta": int(
                report["overall"]["yaw_entry_feasible_control_blocked_rows"] - raw_report["overall"]["yaw_entry_feasible_control_blocked_rows"]
            ),
            "near_basin_shell_yaw_control_observable_rows_delta": int(
                report["overall"]["near_basin_shell_yaw_control_observable_rows"]
                - raw_report["overall"]["near_basin_shell_yaw_control_observable_rows"]
            ),
            "near_basin_shell_yaw_control_blocked_rows_delta": int(
                report["overall"]["near_basin_shell_yaw_control_blocked_rows"]
                - raw_report["overall"]["near_basin_shell_yaw_control_blocked_rows"]
            ),
        }
    report["source_jsonl"] = str(args.relabel_jsonl.resolve())
    if calibration_summary is not None:
        report["yaw_observability_calibration"] = calibration_summary
    if alias_drift_summary is not None:
        report["yaw_observability_alias_drift"] = alias_drift_summary

    out_json = output_dir / "frame_contract_audit.json"
    out_md = output_dir / "frame_contract_audit.md"
    out_json.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    _plot_overview(report, output_dir)

    md_lines = [
        "# Frame Contract Audit",
        "",
        f"- source: `{args.relabel_jsonl}`",
        f"- rows: `{len(rows)}`",
        "",
        "## Overall",
        f"- near_grasp_rate: `{report['overall']['near_grasp_rate']:.3f}`",
        f"- close_ready_rate: `{report['overall']['close_ready_rate']:.3f}`",
        f"- visual_observable_rate: `{report['overall']['visual_observable_rate']:.3f}`",
        f"- prior_only_rate: `{report['overall']['prior_only_rate']:.3f}`",
        f"- micro_entry_ready_rate: `{report['overall']['micro_entry_ready_rate']:.3f}`",
        f"- label_valid_rate: `{report['overall']['label_valid_rate']:.3f}`",
        f"- yaw_entry_feasible_rate: `{report['overall']['yaw_entry_feasible_rate']:.3f}`",
        f"- yaw_control_observable_rate: `{report['overall']['yaw_control_observable_rate']:.3f}`",
        f"- yaw_entry_feasible_control_blocked_rows: `{report['overall']['yaw_entry_feasible_control_blocked_rows']}`",
        f"- yaw_observable_rate: `{report['overall']['yaw_observable_rate']:.3f}`",
        f"- yaw_blocked_rate: `{report['overall']['yaw_blocked_rate']:.3f}`",
        f"- yaw_abstain_correct_rate: `{report['overall']['yaw_abstain_correct_rate']:.3f}`",
        f"- xy_contracted_count: `{report['overall']['xy_contracted_count']}`",
        f"- xy_contraction_lower_ci: `{report['overall']['xy_contraction_lower_ci']:.3f}`",
        f"- visual_observable_yaw_blocked_rows: `{report['overall']['visual_observable_yaw_blocked_rows']}`",
        f"- visual_observable_xy_contracted_yaw_blocked_rows: `{report['overall']['visual_observable_xy_contracted_yaw_blocked_rows']}`",
        f"- visual_observable_xy_contracted_yaw_entry_feasible_control_blocked_rows: `{report['overall']['visual_observable_xy_contracted_yaw_entry_feasible_control_blocked_rows']}`",
        f"- visual_observable_yaw_blocked_rate: `{report['overall']['visual_observable_yaw_blocked_rate']:.3f}`",
        f"- visual_observable_xy_contracted_yaw_blocked_rate: `{report['overall']['visual_observable_xy_contracted_yaw_blocked_rate']:.3f}`",
        f"- coarse_pullback_candidate_rows: `{report['overall']['coarse_pullback_candidate_rows']}`",
        f"- outer_pullback_candidate_rows: `{report['overall']['outer_pullback_candidate_rows']}`",
        f"- near_basin_shell_rows: `{report['overall']['near_basin_shell_rows']}`",
        f"- near_basin_shell_yaw_entry_feasible_rows: `{report['overall']['near_basin_shell_yaw_entry_feasible_rows']}`",
        f"- near_basin_shell_yaw_control_observable_rows: `{report['overall']['near_basin_shell_yaw_control_observable_rows']}`",
        f"- near_basin_shell_yaw_observable_rows: `{report['overall']['near_basin_shell_yaw_observable_rows']}`",
        f"- near_basin_shell_yaw_blocked_rows: `{report['overall']['near_basin_shell_yaw_blocked_rows']}`",
        f"- near_basin_shell_yaw_observable_rate: `{report['overall']['near_basin_shell_yaw_observable_rate']:.3f}`",
        f"- micro_entry_ready_rows: `{report['overall']['micro_entry_ready_rows']}`",
        f"- close_ready_rows: `{report['overall']['close_ready_rows']}`",
        "",
    ]
    if calibration_summary is not None and "comparison" in report:
        md_lines.extend(
            [
                "## Yaw Observability Calibration",
                f"- checkpoint: `{report['yaw_observability_calibration']['checkpoint']}`",
                f"- threshold: `{report['yaw_observability_calibration']['threshold']:.6f}`",
                f"- threshold_source: `{report['yaw_observability_calibration']['threshold_source']}`",
                f"- raw_positive_rows: `{report['yaw_observability_calibration']['raw_positive_rows']}`",
                f"- calibrated_positive_rows: `{report['yaw_observability_calibration']['calibrated_positive_rows']}`",
                f"- delta_positive_rows: `{report['yaw_observability_calibration']['delta_positive_rows']}`",
                f"- mean_probability: `{report['yaw_observability_calibration']['mean_probability']:.3f}`",
                f"- predicted_positive_rate: `{report['yaw_observability_calibration']['predicted_positive_rate']:.3f}`",
                f"- probability_p05_rate: `{report['yaw_observability_calibration']['probability_p05_rate']:.3f}`",
                "",
                "## Calibrated vs Raw",
                f"- yaw_control_observable_rows_delta: `{report['comparison']['yaw_control_observable_rows_delta']}`",
                f"- yaw_control_observable_rate_delta: `{report['comparison']['yaw_control_observable_rate_delta']:.3f}`",
                f"- yaw_entry_feasible_control_blocked_rows_delta: `{report['comparison']['yaw_entry_feasible_control_blocked_rows_delta']}`",
                f"- visual_observable_yaw_blocked_rows_delta: `{report['comparison']['visual_observable_yaw_blocked_rows_delta']}`",
                f"- visual_observable_xy_contracted_yaw_blocked_rows_delta: `{report['comparison']['visual_observable_xy_contracted_yaw_blocked_rows_delta']}`",
                f"- near_basin_shell_yaw_control_observable_rows_delta: `{report['comparison']['near_basin_shell_yaw_control_observable_rows_delta']}`",
                f"- near_basin_shell_yaw_control_blocked_rows_delta: `{report['comparison']['near_basin_shell_yaw_control_blocked_rows_delta']}`",
                f"- near_basin_shell_yaw_observable_rate_delta: `{report['comparison']['near_basin_shell_yaw_observable_rate_delta']:.3f}`",
                "",
            ]
        )
    if alias_drift_summary is not None and "comparison" in report:
        md_lines.extend(
            [
                "## Yaw Alias/Drift Observability",
                f"- rows_jsonl: `{report['yaw_observability_alias_drift']['rows_jsonl']}`",
                f"- matched_rows: `{report['yaw_observability_alias_drift']['matched_rows']}`",
                f"- matched_rate: `{report['yaw_observability_alias_drift']['matched_rate']:.3f}`",
                f"- raw_positive_rows: `{report['yaw_observability_alias_drift']['raw_positive_rows']}`",
                f"- alias_drift_positive_rows: `{report['yaw_observability_alias_drift']['alias_drift_positive_rows']}`",
                f"- alias_drift_rejected_rows: `{report['yaw_observability_alias_drift']['alias_drift_rejected_rows']}`",
                f"- decision_counts: `{report['yaw_observability_alias_drift']['decision_counts']}`",
                f"- truth_role_counts: `{report['yaw_observability_alias_drift']['truth_role_counts']}`",
                "",
                "## Alias/Drift vs Raw",
                f"- yaw_control_observable_rows_delta: `{report['comparison']['yaw_control_observable_rows_delta']}`",
                f"- yaw_control_observable_rate_delta: `{report['comparison']['yaw_control_observable_rate_delta']:.3f}`",
                f"- yaw_entry_feasible_control_blocked_rows_delta: `{report['comparison']['yaw_entry_feasible_control_blocked_rows_delta']}`",
                f"- near_basin_shell_yaw_control_observable_rows_delta: `{report['comparison']['near_basin_shell_yaw_control_observable_rows_delta']}`",
                f"- near_basin_shell_yaw_control_blocked_rows_delta: `{report['comparison']['near_basin_shell_yaw_control_blocked_rows_delta']}`",
                "",
            ]
        )
    md_lines.append("## Axis Summary")
    for axis, stats in report["axis_summary"].items():
        md_lines.append(
            f"- `{axis}`: policy={stats['recommended_policy']}, sign={stats['sign_match_rate']:.3f}, "
            f"contract={stats['contraction_rate']:.3f}, trusted_contract={stats['trusted_contraction_rate']:.3f}, "
            f"proxy_corr={stats['proxy_priv_corr']:.3f}, action_corr={stats['action_priv_corr']:.3f}, "
            f"monotonic={stats['monotonic_prefix_rate']:.3f}, two_step={stats['two_step_monotonic_prefix_rate']:.3f}"
        )
    md_lines.append("")
    md_lines.append("## Yaw Alignment")
    md_lines.append(f"- visual_yaw_crosstab: `{report['overall']['visual_yaw_crosstab']}`")
    md_lines.append(f"- yaw_blocker_combo_counts: `{report['overall']['yaw_blocker_combo_counts']}`")
    md_lines.append(f"- yaw_primary_blocker_counts: `{report['overall']['yaw_primary_blocker_counts']}`")
    md_lines.append(f"- visual_observable_xy_contracted_blocker_combo_counts: `{report['yaw_alignment']['visual_observable_xy_contracted_blocker_combo_counts']}`")
    md_lines.append(f"- visual_observable_yaw_blocked_rows: `{report['overall']['visual_observable_yaw_blocked_rows']}`")
    md_lines.append(f"- visual_observable_xy_contracted_yaw_blocked_rows: `{report['overall']['visual_observable_xy_contracted_yaw_blocked_rows']}`")
    md_lines.append("")
    md_lines.append("## Yaw Observability")
    for item in report["by_yaw_observability"]:
        md_lines.append(
            f"- `{item['yaw_observability_class']}`: rows={item['num_rows']}, tiers={item['takeover_tier_counts']}"
        )
    md_lines.append("")
    md_lines.append("## Visual Summary")
    for item in report["by_visual_observability"]:
        md_lines.append(
            f"- `{item['visual_observability_class']}`: rows={item['axis_summary']['x']['num_rows']}, "
            f"yaw_blocked={item['yaw_blocked_rows']}, visual_yaw_blocked={item['visual_observable_yaw_blocked_rows']}, "
            f"blockers={item['yaw_blocker_combo_counts']}"
        )
    md_lines.append("")
    md_lines.append("## Stage Summary")
    for item in report["by_stage"]:
        md_lines.append(
            f"- `{item['stage_name']}`: rows={item['axis_summary']['x']['num_rows']}, "
            f"yaw_blocked={item['yaw_blocked_rows']}, visual_yaw_blocked={item['visual_observable_yaw_blocked_rows']}, "
            f"blockers={item['yaw_blocker_combo_counts']}"
        )
    md_lines.append("")
    md_lines.append("## Skill Summary")
    for item in report["by_skill"]:
        md_lines.append(
            f"- `{item['skill_name']}`: rows={item['axis_summary']['x']['num_rows']}, "
            f"yaw_blocked={item['yaw_blocked_rows']}, visual_yaw_blocked={item['visual_observable_yaw_blocked_rows']}, "
            f"blockers={item['yaw_blocker_combo_counts']}"
        )
    md_lines.append("")
    md_lines.append("## Takeover Tiers")
    for item in report["by_takeover_tier"]:
        md_lines.append(
            f"- `{item['takeover_tier']}`: rows={item['num_rows']}, xy_contract={item['xy_contraction_rate']:.3f}, "
            f"overshoot={item['overshoot_rate']:.3f}, yaw={item['yaw_observability_counts']}"
        )
        md_lines.append(f"  - xy_contraction_lower_ci: `{item['xy_contraction_lower_ci']:.3f}`")
    md_lines.append("")
    md_lines.append("## Episodes")
    for item in report["by_episode"]:
        md_lines.append(
            f"- `ep{int(item['episode_idx']):03d}`: rows={item['num_rows']}, micro_entry={item['micro_entry_ready_rate']:.3f}, "
            f"tiers={item['takeover_tier_counts']}, yaw={item['yaw_observability_counts']}"
        )
    md_lines.append("")
    md_lines.append("## Window Protocols")
    for item in report["by_window_protocol"]:
        md_lines.append(
            f"- mode={item['window_mode']}, shell={item['shell_filter']}, flush={item['queue_flushed']}, horizon={item['requested_horizon']}: "
            f"rows={item['num_rows']}, micro_entry={item['micro_entry_ready_rate']:.3f}, tiers={item['takeover_tier_counts']}"
        )
    md_lines.append("")
    md_lines.append("## Failure Buckets")
    for item in report["by_failure_bucket"]:
        bucket = item["failure_bucket"]
        md_lines.append(f"- `{bucket}`")
        md_lines.append(f"  - micro_entry_ready_rate: `{item['micro_entry_ready_rate']:.3f}`")
        md_lines.append(f"  - yaw_blocked_rows: `{item['yaw_blocked_rows']}`")
        md_lines.append(f"  - visual_observable_yaw_blocked_rows: `{item['visual_observable_yaw_blocked_rows']}`")
        md_lines.append(f"  - visual_observable_xy_contracted_yaw_blocked_rows: `{item['visual_observable_xy_contracted_yaw_blocked_rows']}`")
        md_lines.append(f"  - yaw_blocker_combo_counts: `{item['yaw_blocker_combo_counts']}`")
        for axis, stats in item["axis_summary"].items():
            md_lines.append(
                f"  - `{axis}`: policy={stats['recommended_policy']}, sign={stats['sign_match_rate']:.3f}, "
                f"contract={stats['contraction_rate']:.3f}, trusted_contract={stats['trusted_contraction_rate']:.3f}, "
                f"proxy_corr={stats['proxy_priv_corr']:.3f}, two_step={stats['two_step_monotonic_prefix_rate']:.3f}"
            )
    out_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print(out_json)
    print(out_md)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
