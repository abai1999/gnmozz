#!/usr/bin/env python3
"""Train the spatial-temporal runtime XY estimator."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from prismatic.robot.coarse2contact_v2.runtime_xy_residual import (
    DEFAULT_RUNTIME_XY_FEATURE_NAMES,
    RUNTIME_XY_SPATIOTEMPORAL_RISK_CLASSES,
    XYSpatialTemporalHeadNet,
    runtime_xy_spatial_temporal_context_feature_vector_from_trace,
    runtime_xy_spatial_temporal_feature_names,
    _load_spatial_temporal_rgbd,
    _spatial_temporal_support_metrics,
)
from prismatic.robot.coarse2contact_v2.xy_spatial_temporal_generalization import (
    SourceRootSplit,
    split_records_by_source_root,
    source_eval_root_key,
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _parse_episode_set(text: str | None) -> set[int]:
    if not text:
        return set()
    out: set[int] = set()
    for item in str(text).split(","):
        item = item.strip()
        if not item:
            continue
        if item.startswith("ep"):
            item = item[2:]
        out.add(int(item))
    return out


def _parse_string_set(text: str | None) -> set[str]:
    if not text:
        return set()
    out: set[str] = set()
    for item in str(text).split(","):
        item = item.strip()
        if item:
            out.add(item)
    return out


def _load_generalization_gate_roots(path: Path | None) -> dict[str, set[str]]:
    if path is None or not str(path):
        return {
            "random10_generalization": set(),
            "random_holdout_pool": set(),
            "sentinel_old4": set(),
            "sentinel_random5": set(),
        }
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    groups = payload.get("groups", payload)
    if not isinstance(groups, Mapping):
        raise ValueError(f"invalid generalization manifest: {path}")
    out: dict[str, set[str]] = {
        "random10_generalization": set(),
        "random_holdout_pool": set(),
        "sentinel_old4": set(),
        "sentinel_random5": set(),
    }
    for group_name in out:
        items = groups.get(group_name, [])
        if not isinstance(items, list):
            continue
        roots = {
            source_eval_root_key(item)
            for item in items
            if isinstance(item, Mapping)
        }
        out[group_name] = {root for root in roots if root}
    return out


def _safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
        if not np.isfinite(out):
            return float(default)
        return out
    except Exception:
        return float(default)


def _safe_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, (int, float, np.integer, np.floating)):
        return bool(float(value) > 0.5)
    return bool(value)


def _sequence_key(row: dict[str, Any]) -> str:
    sequence = str(row.get("sequence_id", "") or "")
    if sequence:
        return sequence
    trace_path = str(row.get("trace_path", "") or "")
    if trace_path:
        return trace_path
    return f"ep{int(row.get('episode_idx', -1)):03d}"


def _group_by_sequence(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    episodes: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        episodes[_sequence_key(row)].append(dict(row))
    for sequence in episodes:
        episodes[sequence].sort(key=lambda r: int(r.get("step_idx", r.get("step", -1))))
    return dict(sorted(episodes.items()))


def _row_weight(
    row: dict[str, Any],
    *,
    include_inactive_rows: bool,
    active_contract_weight: float,
    hard_bucket_weight: float,
    occlusion_weight: float,
    low_observability_weight: float,
    root_balance_weight: float,
) -> float:
    weight = 1.0
    if _safe_bool(row.get("grasp_probe_active"), False):
        weight *= float(active_contract_weight)
    elif include_inactive_rows:
        weight *= 0.4
    bucket = str(row.get("failure_bucket", row.get("bucket", "")) or "unknown")
    if bucket in {"large_xy_large_yaw", "small_xy_large_yaw", "large_xy_small_yaw", "small_xy_small_yaw"}:
        weight *= float(hard_bucket_weight)
    obs = str(row.get("observability_bucket", "") or "")
    if obs == "occluded" or "occlusion" in obs:
        weight *= float(occlusion_weight)
    elif obs in {"low_observability", "low_visibility", "partial_observable", "partial_observation"} or "low" in obs:
        weight *= float(low_observability_weight)
    if float(root_balance_weight) > 0.0:
        weight *= float(root_balance_weight)
    return float(weight)


def _risk_label(row: dict[str, Any], history_rows: list[dict[str, Any]]) -> int:
    if _safe_bool(row.get("wrist_is_occluded"), False) or _safe_bool(row.get("wrist_is_low_visibility"), False):
        return 1
    metrics = _spatial_temporal_support_metrics(row, history_rows, window_size=6)
    if int(metrics["support_rows"]) < 2 or float(metrics["direction_confidence"]) < 0.40:
        return 3
    label = np.asarray(row.get("label_pre_true_error_t", row.get("grasp_probe_pre_true_error_t", [])), dtype=np.float32).reshape(-1)
    proxy = np.asarray(row.get("local_geometry_error", {}).get("grasp", {}).get("dx", 0.0), dtype=np.float32)
    return 2 if _safe_float(proxy, 0.0) * (float(label[0]) if label.size >= 1 else 0.0) < 0.0 else 0


def _bounded_xy_step(pred_xy: torch.Tensor, *, xy_gain: float, max_xy_step: float) -> torch.Tensor:
    step = float(xy_gain) * pred_xy
    norm = torch.linalg.norm(step, dim=-1, keepdim=True)
    if float(max_xy_step) > 0.0:
        scale = torch.ones_like(norm)
        mask = norm > float(max_xy_step)
        scale[mask] = float(max_xy_step) / torch.clamp(norm[mask], min=1.0e-9)
        step = step * scale
    return step


def _selection_score(metrics: dict[str, float]) -> float:
    return (
        1.0 * (1.0 - float(metrics.get("cosine_gt_05_rate", 0.0)))
        + 0.8 * (1.0 - float(metrics.get("sign_match_rate", 0.0)))
        + 0.05 * float(metrics.get("mae", 0.0))
        + 0.45 * (1.0 - float(metrics.get("control_contraction_rate", 0.0)))
        + 0.35 * float(metrics.get("control_worsen_rate", 0.0))
        + 0.75 * float(metrics.get("control_reverse_rate", 0.0))
        + 0.60 * float(metrics.get("control_overshoot_rate", 0.0))
    )


def _worst_case_selection_score(*metric_sets: dict[str, float] | None) -> float:
    scores = [_selection_score(metrics) for metrics in metric_sets if metrics]
    if not scores:
        return float("inf")
    worst = max(scores)
    if len(scores) <= 1:
        return float(worst)
    return float(worst + 0.15 * (worst - min(scores)))


def _step_scale_target_from_xy_error(
    *,
    label_norm: float,
    max_xy_step: float,
    low_visibility: bool,
    support_rows: int,
    recent_support_rows: int,
) -> float:
    if not np.isfinite(float(label_norm)) or float(label_norm) <= 0.0 or float(max_xy_step) <= 0.0:
        return 1.0
    target = float(np.clip(float(label_norm) / float(max_xy_step), 0.0, 1.0))
    if bool(low_visibility):
        target *= 0.80
        if int(recent_support_rows) < 2:
            target *= 0.85
    elif int(support_rows) < 2:
        target *= 0.90
    return float(np.clip(target, 0.05, 1.0))


class SpatialTemporalXYDataset(Dataset):
    def __init__(
        self,
        records: list[dict[str, Any]],
        *,
        window_size: int,
        active_only: bool,
        active_contract_weight: float = 3.0,
        hard_bucket_weight: float = 1.5,
        occlusion_weight: float = 1.75,
        low_observability_weight: float = 1.35,
        root_balance_exponent: float = 0.5,
        max_xy_step: float = 0.003,
    ) -> None:
        self.window_size = int(window_size)
        self.active_only = bool(active_only)
        self.active_contract_weight = float(active_contract_weight)
        self.hard_bucket_weight = float(hard_bucket_weight)
        self.occlusion_weight = float(occlusion_weight)
        self.low_observability_weight = float(low_observability_weight)
        self.root_balance_exponent = float(root_balance_exponent)
        self.max_xy_step = float(max_xy_step)
        self.episodes = _group_by_sequence(records)
        self.root_counts: Counter[str] = Counter()
        for rows in self.episodes.values():
            if not rows:
                continue
            self.root_counts[source_eval_root_key(rows[0])] += len(rows)
        self.samples: list[tuple[int, int]] = []
        for sequence_idx, (_sequence, rows) in enumerate(self.episodes.items()):
            for idx, row in enumerate(rows):
                if self.active_only and not _safe_bool(row.get("grasp_probe_active"), False):
                    continue
                if not _safe_bool(row.get("label_available"), False):
                    continue
                self.samples.append((sequence_idx, idx))
        self.sequence_keys = list(self.episodes.keys())
        self._obs_cache: dict[str, dict[str, np.ndarray]] = {}
        self._image_cache: dict[tuple[int, int], torch.Tensor] = {}

    def __len__(self) -> int:
        return len(self.samples)

    def _load_obs(self, path: str) -> dict[str, np.ndarray]:
        if path not in self._obs_cache:
            with np.load(path, allow_pickle=False) as npz:
                self._obs_cache[path] = {
                    "wrist_rgb": np.asarray(npz["wrist_rgb"], dtype=np.uint8),
                    "wrist_depth": np.asarray(npz["wrist_depth"], dtype=np.float32),
                    "gripper_pose": np.asarray(npz["gripper_pose"], dtype=np.float32),
                }
        return self._obs_cache[path]

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sequence_idx, row_idx = self.samples[idx]
        rows = self.episodes[self.sequence_keys[sequence_idx]]
        row = rows[row_idx]
        history_rows = list(reversed(rows[max(0, row_idx - (self.window_size - 1)) : row_idx]))
        obs_path = str(row.get("runtime_obs_path", row.get("npz_path", "")))
        obs_npz = self._load_obs(obs_path)
        step_idx = int(row["step_idx"])
        robot_state = {
            "planner_delta_7d": np.asarray(row.get("planner_prior_world_6d", row.get("planner_prior_world", row.get("planner_prior_delta", []))), dtype=np.float32).reshape(-1)[:6].tolist(),
            "proprio": np.asarray(row.get("proprio", []), dtype=np.float32).reshape(-1).tolist(),
        }
        cache_key = (int(sequence_idx), int(row_idx))
        rgbd = self._image_cache.get(cache_key)
        if rgbd is None:
            observation = {
                "wrist_rgb": np.asarray(obs_npz["wrist_rgb"][step_idx], dtype=np.uint8),
                "wrist_depth": np.asarray(obs_npz["wrist_depth"][step_idx], dtype=np.float32),
                "gripper_pose": np.asarray(obs_npz["gripper_pose"][step_idx], dtype=np.float32),
            }
            rgbd = _load_spatial_temporal_rgbd(
                observation,
                robot_state,
                crop_size=int(row.get("roi_box", [0, 0, 0, 0])[2] - row.get("roi_box", [0, 0, 0, 0])[0]) if row.get("roi_box") else 96,
                resize_size=96,
            )
            if rgbd is None:
                rgbd = torch.zeros((7, 96, 96), dtype=torch.float32)
            self._image_cache[cache_key] = rgbd
        history_features = runtime_xy_spatial_temporal_context_feature_vector_from_trace(
            row,
            history_rows=history_rows,
            base_feature_names=DEFAULT_RUNTIME_XY_FEATURE_NAMES,
            window_size=self.window_size,
        )
        label = np.asarray(row.get("label_pre_true_error_t", row.get("grasp_probe_pre_true_error_t", [])), dtype=np.float32).reshape(-1)
        if label.size < 2:
            label = np.zeros((2,), dtype=np.float32)
        else:
            label = label[:2]
        risk = _risk_label(row, history_rows)
        support_metrics = _spatial_temporal_support_metrics(row, history_rows, window_size=self.window_size)
        step_scale_target = _step_scale_target_from_xy_error(
            label_norm=float(np.linalg.norm(label[:2])),
            max_xy_step=float(self.max_xy_step),
            low_visibility=bool(support_metrics["low_visibility"]),
            support_rows=int(support_metrics["support_rows"]),
            recent_support_rows=int(support_metrics["recent_support_rows"]),
        )
        root_key = source_eval_root_key(row)
        root_count = max(1, int(self.root_counts.get(root_key, 1)))
        root_balance_weight = float(root_count) ** (-float(self.root_balance_exponent)) if float(self.root_balance_exponent) > 0.0 else 1.0
        return {
            "image": rgbd,
            "history": torch.from_numpy(history_features).float(),
            "proprio": torch.from_numpy(np.asarray(row.get("proprio", [0.0] * 15), dtype=np.float32).reshape(-1)[:15]).float(),
            "planner_prior": torch.from_numpy(np.asarray(row.get("planner_prior_local_6d", [0.0] * 6), dtype=np.float32).reshape(-1)[:6]).float(),
            "label": torch.from_numpy(label.astype(np.float32)),
            "sample_weight": torch.tensor(
                _row_weight(
                    row,
                    include_inactive_rows=not self.active_only,
                    active_contract_weight=self.active_contract_weight,
                    hard_bucket_weight=self.hard_bucket_weight,
                    occlusion_weight=self.occlusion_weight,
                    low_observability_weight=self.low_observability_weight,
                    root_balance_weight=root_balance_weight,
                ),
                dtype=torch.float32,
            ),
            "direction_target": torch.tensor(1.0 if _safe_bool(row.get("grasp_probe_active"), False) else 0.0, dtype=torch.float32),
            "visible_target": torch.tensor(0.0 if _safe_bool(row.get("wrist_is_occluded"), False) or _safe_bool(row.get("wrist_is_low_visibility"), False) else 1.0, dtype=torch.float32),
            "step_scale_target": torch.tensor(step_scale_target, dtype=torch.float32),
            "risk_target": torch.tensor(risk, dtype=torch.long),
            "history_rows": history_rows,
            "row": row,
        }


def _collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "image": torch.stack([item["image"] for item in batch], dim=0),
        "history": torch.stack([item["history"] for item in batch], dim=0),
        "proprio": torch.stack([item["proprio"] for item in batch], dim=0),
        "planner_prior": torch.stack([item["planner_prior"] for item in batch], dim=0),
        "label": torch.stack([item["label"] for item in batch], dim=0),
        "sample_weight": torch.stack([item["sample_weight"] for item in batch], dim=0),
        "direction_target": torch.stack([item["direction_target"] for item in batch], dim=0),
        "visible_target": torch.stack([item["visible_target"] for item in batch], dim=0),
        "step_scale_target": torch.stack([item["step_scale_target"] for item in batch], dim=0),
        "risk_target": torch.stack([item["risk_target"] for item in batch], dim=0),
        "rows": [item["row"] for item in batch],
    }


def _load_records(paths: list[Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in paths:
        if path.is_dir():
            for candidate in sorted(path.glob("*.jsonl")):
                records.extend(_read_jsonl(candidate))
        else:
            records.extend(_read_jsonl(path))
    records.sort(key=lambda r: (_sequence_key(r), int(r.get("step_idx", r.get("step", -1)))))
    return records


def _split_episodes(records: list[dict[str, Any]], val_fraction: float, explicit_val_eps: set[int]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    episodes = sorted({int(r.get("episode_idx", -1)) for r in records})
    if explicit_val_eps:
        val_eps = {ep for ep in explicit_val_eps if ep in episodes}
        train_eps = {ep for ep in episodes if ep not in val_eps}
        return [r for r in records if int(r.get("episode_idx", -1)) in train_eps], [r for r in records if int(r.get("episode_idx", -1)) in val_eps]
    n_val = max(1, int(round(len(episodes) * float(val_fraction))))
    val_eps = set(episodes[-n_val:])
    train_eps = set(episodes[:-n_val])
    return [r for r in records if int(r.get("episode_idx", -1)) in train_eps], [r for r in records if int(r.get("episode_idx", -1)) in val_eps]


def _split_records(
    records: list[dict[str, Any]],
    *,
    split_mode: str,
    val_fraction: float,
    test_fraction: float,
    seed: int,
    explicit_train_roots: set[str],
    explicit_val_roots: set[str],
    explicit_test_roots: set[str],
    explicit_val_eps: set[int],
) -> SourceRootSplit:
    if split_mode == "episode":
        train_records, val_records = _split_episodes(records, val_fraction, explicit_val_eps)
        return SourceRootSplit(
            split_mode="episode",
            train_records=train_records,
            val_records=val_records,
            test_records=[],
            train_source_eval_roots=sorted({source_eval_root_key(r) for r in train_records}),
            val_source_eval_roots=sorted({source_eval_root_key(r) for r in val_records}),
            test_source_eval_roots=[],
        )
    return split_records_by_source_root(
        records,
        split_mode=split_mode,
        val_fraction=val_fraction,
        test_fraction=test_fraction,
        seed=seed,
        train_source_eval_roots=explicit_train_roots,
        val_source_eval_roots=explicit_val_roots,
        test_source_eval_roots=explicit_test_roots,
    )


@torch.no_grad()
def _evaluate(
    model: XYSpatialTemporalHeadNet,
    loader: DataLoader,
    device: torch.device,
    *,
    xy_gain: float,
    max_xy_step: float,
) -> dict[str, float]:
    model.eval()
    all_cos = []
    all_sign = []
    all_mae = []
    all_control_cos = []
    all_control_sign = []
    all_contraction = []
    all_worsen = []
    all_overshoot = []
    all_reverse = []
    all_visible = []
    all_step_scale = []
    all_risk_pred = []
    all_risk_tgt = []
    for batch in loader:
        out = model(
            batch["image"].to(device),
            batch["history"].to(device),
            batch["proprio"].to(device),
            batch["planner_prior"].to(device),
        )
        label = batch["label"].to(device)
        pred_xy = torch.stack([out["dx"], out["dy"]], dim=-1)
        step = _bounded_xy_step(pred_xy, xy_gain=xy_gain, max_xy_step=max_xy_step) * out["xy_step_scale"].unsqueeze(-1)
        post = label - step
        pre_norm = torch.linalg.norm(label, dim=-1)
        post_norm = torch.linalg.norm(post, dim=-1)
        step_norm = torch.linalg.norm(step, dim=-1)
        cos = F.cosine_similarity(pred_xy, label, dim=-1, eps=1.0e-6)
        control_cos = F.cosine_similarity(step, label, dim=-1, eps=1.0e-6)
        sign = (torch.sign(pred_xy) == torch.sign(label)).float().mean(dim=-1)
        control_sign = (torch.sign(step) == torch.sign(label)).float().mean(dim=-1)
        risk_pred = torch.argmax(out["risk_logits"], dim=-1)
        all_cos.extend(cos.detach().cpu().tolist())
        all_sign.extend(sign.detach().cpu().tolist())
        all_mae.extend(torch.mean(torch.abs(pred_xy - label), dim=-1).detach().cpu().tolist())
        all_control_cos.extend(control_cos.detach().cpu().tolist())
        all_control_sign.extend(control_sign.detach().cpu().tolist())
        all_contraction.extend((post_norm < pre_norm).float().detach().cpu().tolist())
        all_worsen.extend((post_norm > pre_norm).float().detach().cpu().tolist())
        all_overshoot.extend((step_norm > pre_norm).float().detach().cpu().tolist())
        all_reverse.extend((F.cosine_similarity(step, label, dim=-1, eps=1.0e-6) < 0.0).float().detach().cpu().tolist())
        all_visible.extend(out["xy_visible_confidence"].detach().cpu().tolist())
        all_step_scale.extend(out["xy_step_scale"].detach().cpu().tolist())
        all_risk_pred.extend(risk_pred.detach().cpu().tolist())
        all_risk_tgt.extend(batch["risk_target"].detach().cpu().tolist())
    all_cos = np.asarray(all_cos, dtype=np.float32)
    all_sign = np.asarray(all_sign, dtype=np.float32)
    all_mae = np.asarray(all_mae, dtype=np.float32)
    all_control_cos = np.asarray(all_control_cos, dtype=np.float32)
    all_control_sign = np.asarray(all_control_sign, dtype=np.float32)
    all_contraction = np.asarray(all_contraction, dtype=np.float32)
    all_worsen = np.asarray(all_worsen, dtype=np.float32)
    all_overshoot = np.asarray(all_overshoot, dtype=np.float32)
    all_reverse = np.asarray(all_reverse, dtype=np.float32)
    all_visible = np.asarray(all_visible, dtype=np.float32)
    all_step_scale = np.asarray(all_step_scale, dtype=np.float32)
    all_risk_pred = np.asarray(all_risk_pred, dtype=np.int64)
    all_risk_tgt = np.asarray(all_risk_tgt, dtype=np.int64)
    return {
        "rows": int(all_cos.size),
        "cosine_mean": float(np.mean(all_cos)) if all_cos.size else 0.0,
        "cosine_gt_05_rate": float(np.mean(all_cos > 0.5)) if all_cos.size else 0.0,
        "sign_match_rate": float(np.mean(all_sign)) if all_sign.size else 0.0,
        "mae": float(np.mean(all_mae)) if all_mae.size else 0.0,
        "control_cosine_mean": float(np.mean(all_control_cos)) if all_control_cos.size else 0.0,
        "control_sign_match_rate": float(np.mean(all_control_sign)) if all_control_sign.size else 0.0,
        "control_contraction_rate": float(np.mean(all_contraction)) if all_contraction.size else 0.0,
        "control_worsen_rate": float(np.mean(all_worsen)) if all_worsen.size else 0.0,
        "control_overshoot_rate": float(np.mean(all_overshoot)) if all_overshoot.size else 0.0,
        "control_reverse_rate": float(np.mean(all_reverse)) if all_reverse.size else 0.0,
        "xy_visible_confidence_mean": float(np.mean(all_visible)) if all_visible.size else 0.0,
        "xy_step_scale_mean": float(np.mean(all_step_scale)) if all_step_scale.size else 0.0,
        "risk_accuracy": float(np.mean(all_risk_pred == all_risk_tgt)) if all_risk_pred.size else 0.0,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_jsonl", type=Path, nargs="+", required=True)
    ap.add_argument("--output_checkpoint", type=Path, default=Path("runtime_artifacts/coarse2contact_v2/checkpoints/runtime_xy_spatial_temporal_v42_generalization_candidate.pt"))
    ap.add_argument("--output_json", type=Path, default=Path("runtime_artifacts/coarse2contact_v2/reports/runtime_xy_spatial_temporal_v42_generalization_candidate_train.json"))
    ap.add_argument("--output_md", type=Path, default=Path("runtime_artifacts/coarse2contact_v2/reports/runtime_xy_spatial_temporal_v42_generalization_candidate_train.md"))
    ap.add_argument("--split_mode", type=str, default="auto", choices=("auto", "root", "episode"))
    ap.add_argument("--val_fraction", type=float, default=0.15)
    ap.add_argument("--test_fraction", type=float, default=0.15)
    ap.add_argument("--val_source_eval_roots", type=str, default="")
    ap.add_argument("--test_source_eval_roots", type=str, default="")
    ap.add_argument("--train_source_eval_roots", type=str, default="")
    ap.add_argument("--generalization_manifest_json", type=Path, default=None)
    ap.add_argument("--init_checkpoint", type=Path, default=None)
    ap.add_argument("--val_episodes", type=str, default="")
    ap.add_argument("--window_size", type=int, default=6)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2.0e-4)
    ap.add_argument("--weight_decay", type=float, default=1.0e-4)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--xy_gain", type=float, default=0.35)
    ap.add_argument("--max_xy_step", type=float, default=0.003)
    ap.add_argument("--direction_weight", type=float, default=1.0)
    ap.add_argument("--sign_weight", type=float, default=0.8)
    ap.add_argument("--mae_weight", type=float, default=0.05)
    ap.add_argument("--contraction_weight", type=float, default=0.45)
    ap.add_argument("--control_reverse_weight", type=float, default=0.75)
    ap.add_argument("--active_contract_weight", type=float, default=3.0)
    ap.add_argument("--hard_bucket_weight", type=float, default=1.5)
    ap.add_argument("--occlusion_weight", type=float, default=1.75)
    ap.add_argument("--low_observability_weight", type=float, default=1.35)
    ap.add_argument("--root_balance_exponent", type=float, default=0.5)
    ap.add_argument("--control_overshoot_weight", type=float, default=1.0)
    ap.add_argument("--include_inactive_rows", action="store_true", default=False)
    ap.add_argument("--num_workers", type=int, default=max(0, min(8, (os.cpu_count() or 4) // 2)))
    ap.add_argument("--pin_memory", action="store_true", default=True)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    torch.cuda.manual_seed_all(int(args.seed))

    records = _load_records([Path(p) for p in args.dataset_jsonl])
    generalization_roots = _load_generalization_gate_roots(args.generalization_manifest_json)
    gate_roots = set(generalization_roots.get("random10_generalization", set()))
    holdout_roots = set(generalization_roots.get("random_holdout_pool", set()))
    sentinel_old4_roots = set(generalization_roots.get("sentinel_old4", set()))
    sentinel_random5_roots = set(generalization_roots.get("sentinel_random5", set()))
    gate_records = [r for r in records if source_eval_root_key(r) in gate_roots]
    holdout_pool_records = [r for r in records if source_eval_root_key(r) in holdout_roots]
    sentinel_old4_records = [r for r in records if source_eval_root_key(r) in sentinel_old4_roots]
    sentinel_random5_records = [r for r in records if source_eval_root_key(r) in sentinel_random5_roots]
    if gate_roots:
        records = [r for r in records if source_eval_root_key(r) not in gate_roots]
    explicit_val_eps = _parse_episode_set(args.val_episodes)
    explicit_train_roots = _parse_string_set(args.train_source_eval_roots)
    explicit_val_roots = _parse_string_set(args.val_source_eval_roots)
    explicit_test_roots = _parse_string_set(args.test_source_eval_roots)
    split = _split_records(
        records,
        split_mode=str(args.split_mode),
        val_fraction=float(args.val_fraction),
        test_fraction=float(args.test_fraction),
        seed=int(args.seed),
        explicit_train_roots=explicit_train_roots,
        explicit_val_roots=explicit_val_roots,
        explicit_test_roots=explicit_test_roots,
        explicit_val_eps=explicit_val_eps,
    )
    train_records = split.train_records
    val_records = split.val_records
    test_records = split.test_records
    if not train_records or not val_records:
        raise RuntimeError("Need both train and val records")
    history_feature_names = runtime_xy_spatial_temporal_feature_names(DEFAULT_RUNTIME_XY_FEATURE_NAMES, int(args.window_size))
    feature_names = list(DEFAULT_RUNTIME_XY_FEATURE_NAMES)
    train_ds = SpatialTemporalXYDataset(
        train_records,
        window_size=int(args.window_size),
        active_only=not bool(args.include_inactive_rows),
        active_contract_weight=float(args.active_contract_weight),
        hard_bucket_weight=float(args.hard_bucket_weight),
        occlusion_weight=float(args.occlusion_weight),
        low_observability_weight=float(args.low_observability_weight),
        root_balance_exponent=float(args.root_balance_exponent),
        max_xy_step=float(args.max_xy_step),
    )
    val_ds = SpatialTemporalXYDataset(
        val_records,
        window_size=int(args.window_size),
        active_only=not bool(args.include_inactive_rows),
        active_contract_weight=float(args.active_contract_weight),
        hard_bucket_weight=float(args.hard_bucket_weight),
        occlusion_weight=float(args.occlusion_weight),
        low_observability_weight=float(args.low_observability_weight),
        root_balance_exponent=float(args.root_balance_exponent),
        max_xy_step=float(args.max_xy_step),
    )
    test_ds = SpatialTemporalXYDataset(
        test_records,
        window_size=int(args.window_size),
        active_only=not bool(args.include_inactive_rows),
        active_contract_weight=float(args.active_contract_weight),
        hard_bucket_weight=float(args.hard_bucket_weight),
        occlusion_weight=float(args.occlusion_weight),
        low_observability_weight=float(args.low_observability_weight),
        root_balance_exponent=float(args.root_balance_exponent),
        max_xy_step=float(args.max_xy_step),
    ) if test_records else None
    gate_ds = SpatialTemporalXYDataset(
        gate_records,
        window_size=int(args.window_size),
        active_only=not bool(args.include_inactive_rows),
        active_contract_weight=float(args.active_contract_weight),
        hard_bucket_weight=float(args.hard_bucket_weight),
        occlusion_weight=float(args.occlusion_weight),
        low_observability_weight=float(args.low_observability_weight),
        root_balance_exponent=float(args.root_balance_exponent),
        max_xy_step=float(args.max_xy_step),
    ) if gate_records else None
    holdout_pool_ds = SpatialTemporalXYDataset(
        holdout_pool_records,
        window_size=int(args.window_size),
        active_only=not bool(args.include_inactive_rows),
        active_contract_weight=float(args.active_contract_weight),
        hard_bucket_weight=float(args.hard_bucket_weight),
        occlusion_weight=float(args.occlusion_weight),
        low_observability_weight=float(args.low_observability_weight),
        root_balance_exponent=float(args.root_balance_exponent),
        max_xy_step=float(args.max_xy_step),
    ) if holdout_pool_records else None
    sentinel_old4_ds = SpatialTemporalXYDataset(
        sentinel_old4_records,
        window_size=int(args.window_size),
        active_only=not bool(args.include_inactive_rows),
        active_contract_weight=float(args.active_contract_weight),
        hard_bucket_weight=float(args.hard_bucket_weight),
        occlusion_weight=float(args.occlusion_weight),
        low_observability_weight=float(args.low_observability_weight),
        root_balance_exponent=float(args.root_balance_exponent),
        max_xy_step=float(args.max_xy_step),
    ) if sentinel_old4_records else None
    sentinel_random5_ds = SpatialTemporalXYDataset(
        sentinel_random5_records,
        window_size=int(args.window_size),
        active_only=not bool(args.include_inactive_rows),
        active_contract_weight=float(args.active_contract_weight),
        hard_bucket_weight=float(args.hard_bucket_weight),
        occlusion_weight=float(args.occlusion_weight),
        low_observability_weight=float(args.low_observability_weight),
        root_balance_exponent=float(args.root_balance_exponent),
        max_xy_step=float(args.max_xy_step),
    ) if sentinel_random5_records else None
    loader_kwargs = {
        "batch_size": int(args.batch_size),
        "num_workers": max(0, int(args.num_workers)),
        "pin_memory": bool(args.pin_memory) and str(args.device).startswith("cuda"),
        "persistent_workers": bool(int(args.num_workers) > 0),
    }
    if int(args.num_workers) <= 0:
        loader_kwargs.pop("persistent_workers", None)
    train_loader = DataLoader(train_ds, shuffle=True, collate_fn=_collate, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, collate_fn=_collate, **loader_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, collate_fn=_collate, **loader_kwargs) if test_ds is not None else None
    gate_loader = DataLoader(gate_ds, shuffle=False, collate_fn=_collate, **loader_kwargs) if gate_ds is not None else None
    holdout_pool_loader = DataLoader(holdout_pool_ds, shuffle=False, collate_fn=_collate, **loader_kwargs) if holdout_pool_ds is not None else None
    sentinel_old4_loader = DataLoader(sentinel_old4_ds, shuffle=False, collate_fn=_collate, **loader_kwargs) if sentinel_old4_ds is not None else None
    sentinel_random5_loader = DataLoader(sentinel_random5_ds, shuffle=False, collate_fn=_collate, **loader_kwargs) if sentinel_random5_ds is not None else None
    model = XYSpatialTemporalHeadNet(
        image_in_channels=7,
        image_hidden_dim=128,
        history_feature_dim=max(1, len(history_feature_names) // int(args.window_size)),
        history_window_size=int(args.window_size),
        proprio_dim=15,
        planner_prior_dim=6,
        risk_classes=RUNTIME_XY_SPATIOTEMPORAL_RISK_CLASSES,
    ).to(args.device)
    if args.init_checkpoint is not None and str(args.init_checkpoint):
        init_state = torch.load(args.init_checkpoint, map_location="cpu")
        if not isinstance(init_state, Mapping) or "model_state_dict" not in init_state:
            raise ValueError(f"invalid init checkpoint: {args.init_checkpoint}")
        model.load_state_dict(init_state["model_state_dict"])
    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    best_score = float("inf")
    best_epoch = 0 if args.init_checkpoint is not None and str(args.init_checkpoint) else -1
    history: list[dict[str, Any]] = []
    eps = 1.0e-6
    if args.init_checkpoint is not None and str(args.init_checkpoint):
        init_val_metrics = _evaluate(model, val_loader, torch.device(args.device), xy_gain=float(args.xy_gain), max_xy_step=float(args.max_xy_step))
        init_gate_metrics = _evaluate(model, gate_loader, torch.device(args.device), xy_gain=float(args.xy_gain), max_xy_step=float(args.max_xy_step)) if gate_loader is not None else {}
        init_holdout_pool_metrics = _evaluate(model, holdout_pool_loader, torch.device(args.device), xy_gain=float(args.xy_gain), max_xy_step=float(args.max_xy_step)) if holdout_pool_loader is not None else {}
        init_sentinel_old4_metrics = _evaluate(model, sentinel_old4_loader, torch.device(args.device), xy_gain=float(args.xy_gain), max_xy_step=float(args.max_xy_step)) if sentinel_old4_loader is not None else {}
        init_sentinel_random5_metrics = _evaluate(model, sentinel_random5_loader, torch.device(args.device), xy_gain=float(args.xy_gain), max_xy_step=float(args.max_xy_step)) if sentinel_random5_loader is not None else {}
        best_score = float(_worst_case_selection_score(
            init_val_metrics,
            init_gate_metrics,
            init_holdout_pool_metrics,
            init_sentinel_old4_metrics,
            init_sentinel_random5_metrics,
        ))
        history.append({
            "epoch": 0,
            "score": float(best_score),
            "val": init_val_metrics,
            "gate": init_gate_metrics,
            "holdout_pool": init_holdout_pool_metrics,
            "sentinel_old4": init_sentinel_old4_metrics,
            "sentinel_random5": init_sentinel_random5_metrics,
        })
    for epoch in range(int(args.epochs)):
        model.train()
        for batch in train_loader:
            image = batch["image"].to(args.device)
            history_features = batch["history"].to(args.device)
            proprio = batch["proprio"].to(args.device)
            planner_prior = batch["planner_prior"].to(args.device)
            label = batch["label"].to(args.device)
            sample_weight = batch["sample_weight"].to(args.device)
            direction_target = batch["direction_target"].to(args.device)
            visible_target = batch["visible_target"].to(args.device)
            step_scale_target = batch["step_scale_target"].to(args.device)
            risk_target = batch["risk_target"].to(args.device)

            out = model(image, history_features, proprio, planner_prior)
            pred_xy = torch.stack([out["dx"], out["dy"]], dim=-1)
            step = _bounded_xy_step(pred_xy, xy_gain=float(args.xy_gain), max_xy_step=float(args.max_xy_step)) * out["xy_step_scale"].unsqueeze(-1)
            post = label - step
            pre_norm = torch.linalg.norm(label, dim=-1)
            post_norm = torch.linalg.norm(post, dim=-1)
            step_norm = torch.linalg.norm(step, dim=-1)
            cosine = F.cosine_similarity(pred_xy, label, dim=-1, eps=eps)
            control_cosine = F.cosine_similarity(step, label, dim=-1, eps=eps)
            sign_match = (torch.sign(pred_xy) == torch.sign(label)).float().mean(dim=-1)
            control_sign_match = (torch.sign(step) == torch.sign(label)).float().mean(dim=-1)
            direction_loss = 0.5 * (1.0 - cosine) + 0.5 * (1.0 - control_cosine)
            sign_loss = 0.5 * (1.0 - sign_match) + 0.5 * (1.0 - control_sign_match)
            mae_loss = 0.5 * torch.mean(torch.abs(pred_xy - label), dim=-1) + 0.5 * torch.mean(torch.abs(step - label), dim=-1)
            contraction_loss = torch.relu(post_norm - pre_norm)
            worsen_loss = torch.relu(post_norm - pre_norm)
            overshoot_loss = torch.relu(step_norm - pre_norm)
            reverse_loss = torch.relu(-control_cosine)
            visible_loss = F.binary_cross_entropy(out["xy_visible_confidence"], visible_target, reduction="none")
            step_scale_loss = F.mse_loss(out["xy_step_scale"], step_scale_target, reduction="none")
            direction_conf_target = torch.clamp(((cosine > 0.25) & (control_cosine > 0.25)).float(), 0.0, 1.0)
            direction_conf_loss = F.binary_cross_entropy(out["xy_direction_confidence"], direction_conf_target, reduction="none")
            risk_loss = F.cross_entropy(out["risk_logits"], risk_target, reduction="none")
            loss_row = (
                float(args.direction_weight) * direction_loss
                + float(args.sign_weight) * sign_loss
                + float(args.mae_weight) * mae_loss
                + float(args.contraction_weight) * contraction_loss
                + 0.5 * float(args.contraction_weight) * worsen_loss
                + float(args.control_reverse_weight) * reverse_loss
                + float(args.control_overshoot_weight) * overshoot_loss
                + 0.2 * visible_loss
                + 0.8 * step_scale_loss
                + 0.1 * direction_conf_loss
                + 0.1 * risk_loss
            )
            loss = torch.sum(loss_row * sample_weight) / torch.clamp(torch.sum(sample_weight), min=eps)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            opt.step()

        if epoch == 0 or epoch == int(args.epochs) - 1 or (epoch + 1) % max(1, int(args.epochs) // 5) == 0:
            val_metrics = _evaluate(model, val_loader, torch.device(args.device), xy_gain=float(args.xy_gain), max_xy_step=float(args.max_xy_step))
            gate_metrics = _evaluate(model, gate_loader, torch.device(args.device), xy_gain=float(args.xy_gain), max_xy_step=float(args.max_xy_step)) if gate_loader is not None else {}
            holdout_pool_metrics = _evaluate(model, holdout_pool_loader, torch.device(args.device), xy_gain=float(args.xy_gain), max_xy_step=float(args.max_xy_step)) if holdout_pool_loader is not None else {}
            sentinel_old4_metrics = _evaluate(model, sentinel_old4_loader, torch.device(args.device), xy_gain=float(args.xy_gain), max_xy_step=float(args.max_xy_step)) if sentinel_old4_loader is not None else {}
            sentinel_random5_metrics = _evaluate(model, sentinel_random5_loader, torch.device(args.device), xy_gain=float(args.xy_gain), max_xy_step=float(args.max_xy_step)) if sentinel_random5_loader is not None else {}
            val_score = _selection_score(val_metrics)
            gate_score = _selection_score(gate_metrics) if gate_metrics else float("inf")
            holdout_score = _selection_score(holdout_pool_metrics) if holdout_pool_metrics else float("inf")
            sentinel_old4_score = _selection_score(sentinel_old4_metrics) if sentinel_old4_metrics else float("inf")
            sentinel_random5_score = _selection_score(sentinel_random5_metrics) if sentinel_random5_metrics else float("inf")
            score = _worst_case_selection_score(
                val_metrics,
                gate_metrics,
                holdout_pool_metrics,
                sentinel_old4_metrics,
                sentinel_random5_metrics,
            )
            history.append({
                "epoch": int(epoch + 1),
                "score": float(score),
                "val_score": float(val_score),
                "gate_score": float(gate_score) if np.isfinite(gate_score) else None,
                "holdout_score": float(holdout_score) if np.isfinite(holdout_score) else None,
                "val": val_metrics,
                "gate": gate_metrics,
                "holdout_pool": holdout_pool_metrics,
                "sentinel_old4_score": float(sentinel_old4_score) if np.isfinite(sentinel_old4_score) else None,
                "sentinel_random5_score": float(sentinel_random5_score) if np.isfinite(sentinel_random5_score) else None,
                "sentinel_old4": sentinel_old4_metrics,
                "sentinel_random5": sentinel_random5_metrics,
            })
            print(
                json.dumps(
                    {
                        "epoch": int(epoch + 1),
                        "score": float(score),
                        "val_score": float(val_score),
                        "gate_score": float(gate_score) if np.isfinite(gate_score) else None,
                        "holdout_score": float(holdout_score) if np.isfinite(holdout_score) else None,
                        "val_contraction": float(val_metrics["control_contraction_rate"]),
                        "val_worsen": float(val_metrics["control_worsen_rate"]),
                        "val_overshoot": float(val_metrics["control_overshoot_rate"]),
                        "val_reverse": float(val_metrics["control_reverse_rate"]),
                        "gate_contraction": float(gate_metrics["control_contraction_rate"]) if gate_metrics else None,
                        "gate_worsen": float(gate_metrics["control_worsen_rate"]) if gate_metrics else None,
                        "gate_overshoot": float(gate_metrics["control_overshoot_rate"]) if gate_metrics else None,
                        "gate_reverse": float(gate_metrics["control_reverse_rate"]) if gate_metrics else None,
                        "sentinel_old4_reverse": float(sentinel_old4_metrics["control_reverse_rate"]) if sentinel_old4_metrics else None,
                        "sentinel_random5_reverse": float(sentinel_random5_metrics["control_reverse_rate"]) if sentinel_random5_metrics else None,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            if score < best_score:
                best_score = float(score)
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                best_epoch = int(epoch + 1)
                checkpoint = {
                    "schema_version": "c2c_v2_runtime_xy_spatial_temporal_checkpoint_v1",
                    "model_type": "spatial_temporal",
                    "config": {
                        "feature_names": feature_names,
                        "history_feature_names": list(history_feature_names),
                        "history_window_size": int(args.window_size),
                        "image_in_channels": 7,
                        "image_hidden_dim": 128,
                        "image_crop_size": 96,
                        "image_resize_size": 96,
                        "proprio_dim": 15,
                        "planner_prior_dim": 6,
                        "risk_classes": list(RUNTIME_XY_SPATIOTEMPORAL_RISK_CLASSES),
                    },
                    "model_state_dict": {k: v.detach().cpu() for k, v in best_state.items()},
                    "source": "runtime_xy_spatial_temporal_v42_generalization_candidate",
                }
                args.output_checkpoint.parent.mkdir(parents=True, exist_ok=True)
                torch.save(checkpoint, args.output_checkpoint)

    if best_state is not None:
        model.load_state_dict(best_state)
    checkpoint = {
        "schema_version": "c2c_v2_runtime_xy_spatial_temporal_checkpoint_v1",
        "model_type": "spatial_temporal",
        "config": {
            "feature_names": feature_names,
            "history_feature_names": list(history_feature_names),
            "history_window_size": int(args.window_size),
            "image_in_channels": 7,
            "image_hidden_dim": 128,
            "image_crop_size": 96,
            "image_resize_size": 96,
            "proprio_dim": 15,
            "planner_prior_dim": 6,
            "risk_classes": list(RUNTIME_XY_SPATIOTEMPORAL_RISK_CLASSES),
        },
        "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "source": "runtime_xy_spatial_temporal_v42_generalization_candidate",
    }
    args.output_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.output_checkpoint)
    train_loader_eval = DataLoader(train_ds, batch_size=int(args.batch_size), shuffle=False, num_workers=0, collate_fn=_collate)
    train_metrics = _evaluate(model, train_loader_eval, torch.device(args.device), xy_gain=float(args.xy_gain), max_xy_step=float(args.max_xy_step))
    val_metrics = _evaluate(model, val_loader, torch.device(args.device), xy_gain=float(args.xy_gain), max_xy_step=float(args.max_xy_step))
    test_metrics = _evaluate(model, test_loader, torch.device(args.device), xy_gain=float(args.xy_gain), max_xy_step=float(args.max_xy_step)) if test_loader is not None else {}
    gate_metrics = _evaluate(model, gate_loader, torch.device(args.device), xy_gain=float(args.xy_gain), max_xy_step=float(args.max_xy_step)) if gate_loader is not None else {}
    holdout_pool_metrics = _evaluate(model, holdout_pool_loader, torch.device(args.device), xy_gain=float(args.xy_gain), max_xy_step=float(args.max_xy_step)) if holdout_pool_loader is not None else {}
    sentinel_old4_metrics = _evaluate(model, sentinel_old4_loader, torch.device(args.device), xy_gain=float(args.xy_gain), max_xy_step=float(args.max_xy_step)) if sentinel_old4_loader is not None else {}
    sentinel_random5_metrics = _evaluate(model, sentinel_random5_loader, torch.device(args.device), xy_gain=float(args.xy_gain), max_xy_step=float(args.max_xy_step)) if sentinel_random5_loader is not None else {}
    report = {
        "schema_version": "c2c_v2_runtime_xy_spatial_temporal_train_v1",
        "model_type": "spatial_temporal",
        "rows": int(len(records)),
        "train_rows": int(len(train_ds)),
        "val_rows": int(len(val_ds)),
        "test_rows": int(len(test_ds)) if test_ds is not None else 0,
        "gate_rows": int(len(gate_ds)) if gate_ds is not None else 0,
        "holdout_pool_rows": int(len(holdout_pool_ds)) if holdout_pool_ds is not None else 0,
        "best_epoch": int(best_epoch),
        "train_episodes": sorted({int(r.get("episode_idx", -1)) for r in train_records}),
        "val_episodes": sorted({int(r.get("episode_idx", -1)) for r in val_records}),
        "test_episodes": sorted({int(r.get("episode_idx", -1)) for r in test_records}),
        "explicit_val_episodes": sorted(explicit_val_eps),
        "explicit_val_source_eval_roots": sorted(explicit_val_roots),
        "explicit_test_source_eval_roots": sorted(explicit_test_roots),
        "split_mode": split.split_mode,
        "train_source_eval_roots": split.train_source_eval_roots,
        "val_source_eval_roots": split.val_source_eval_roots,
        "test_source_eval_roots": split.test_source_eval_roots,
        "gate_source_eval_roots": sorted(gate_roots),
        "holdout_pool_source_eval_roots": sorted(holdout_roots),
        "generalization_manifest_json": str(args.generalization_manifest_json) if args.generalization_manifest_json else "",
        "init_checkpoint": str(args.init_checkpoint) if args.init_checkpoint else "",
        "history_window_size": int(args.window_size),
        "feature_names": feature_names,
        "history_feature_names": list(history_feature_names),
        "selection_weights": {
            "direction_weight": float(args.direction_weight),
            "sign_weight": float(args.sign_weight),
            "mae_weight": float(args.mae_weight),
            "contraction_weight": float(args.contraction_weight),
            "control_reverse_weight": float(args.control_reverse_weight),
            "control_overshoot_weight": float(args.control_overshoot_weight),
            "active_contract_weight": float(args.active_contract_weight),
            "hard_bucket_weight": float(args.hard_bucket_weight),
            "occlusion_weight": float(args.occlusion_weight),
            "low_observability_weight": float(args.low_observability_weight),
            "root_balance_exponent": float(args.root_balance_exponent),
        },
        "include_inactive_rows": bool(args.include_inactive_rows),
        "runtime_upgrade_gate": {
            "requires_mp4_runtime_ab": True,
            "requires_hard_bucket_runtime_ab": True,
            "decision": "pending_runtime_ab_validation",
        },
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "gate_metrics": gate_metrics,
        "holdout_pool_metrics": holdout_pool_metrics,
        "sentinel_old4_rows": int(len(sentinel_old4_ds)) if sentinel_old4_ds is not None else 0,
        "sentinel_random5_rows": int(len(sentinel_random5_ds)) if sentinel_random5_ds is not None else 0,
        "sentinel_old4_metrics": sentinel_old4_metrics,
        "sentinel_random5_metrics": sentinel_random5_metrics,
        "risk_classes": list(RUNTIME_XY_SPATIOTEMPORAL_RISK_CLASSES),
        "checkpoint": str(args.output_checkpoint),
        "history": history,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    lines = [
        "# Runtime XY Spatial-Temporal Estimator",
        "",
        f"- rows: `{report['rows']}`",
        f"- train_rows: `{report['train_rows']}`",
        f"- val_rows: `{report['val_rows']}`",
        f"- test_rows: `{report['test_rows']}`",
        f"- train cosine_gt_05: `{train_metrics['cosine_gt_05_rate']:.3f}`",
        f"- val cosine_gt_05: `{val_metrics['cosine_gt_05_rate']:.3f}`",
        f"- val contraction: `{val_metrics['control_contraction_rate']:.3f}`",
        f"- val worsen: `{val_metrics['control_worsen_rate']:.3f}`",
        f"- val overshoot: `{val_metrics['control_overshoot_rate']:.3f}`",
        f"- val reverse: `{val_metrics['control_reverse_rate']:.3f}`",
        f"- gate rows: `{report['gate_rows']}`",
        f"- gate contraction: `{gate_metrics['control_contraction_rate']:.3f}`" if gate_metrics else "- gate contraction: `n/a`",
        f"- gate worsen: `{gate_metrics['control_worsen_rate']:.3f}`" if gate_metrics else "- gate worsen: `n/a`",
        f"- gate overshoot: `{gate_metrics['control_overshoot_rate']:.3f}`" if gate_metrics else "- gate overshoot: `n/a`",
        f"- gate reverse: `{gate_metrics['control_reverse_rate']:.3f}`" if gate_metrics else "- gate reverse: `n/a`",
        f"- split_mode: `{report['split_mode']}`",
        f"- checkpoint: `{args.output_checkpoint}`",
        f"- runtime_upgrade_gate: `{report['runtime_upgrade_gate']['decision']}`",
    ]
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
