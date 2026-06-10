#!/usr/bin/env python3
"""Train a v46 candidate-command ranking head via the command transition output.

This script keeps the runtime input contract unchanged: wrist RGBD/depth
validity/proprio/planner prior/history/candidate command are runtime-visible,
while true pre/post task-frame residuals are offline labels only. The key
difference from plain command-delta training is that loss is computed over
candidate groups from the same episode+step, so the model learns which command
should be selected rather than only regressing each transition independently.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from prismatic.robot.coarse2contact_v2.runtime_xy_residual import (  # noqa: E402
    RUNTIME_XY_SPATIAL_TEMPORAL_CROP_SIZE,
    RUNTIME_XY_SPATIAL_TEMPORAL_HISTORY_WINDOW,
    RUNTIME_XY_SPATIAL_TEMPORAL_IMAGE_SIZE,
)
from prismatic.robot.coarse2contact_v2.task_frame_v46_alignment import (  # noqa: E402
    TASK_FRAME_V46_RISK_CLASSES,
    TaskFrameV46AlignmentNet,
    save_task_frame_v46_alignment_checkpoint,
)
from prismatic.robot.coarse2contact_v2.task_frame_readiness import TASK_FRAME_READINESS_FEATURE_NAMES  # noqa: E402
from prismatic.robot.coarse2contact_v2.xy_spatial_temporal_generalization import source_eval_root_key  # noqa: E402
from prismatic.robot.coarse2contact_v2.xy_spatial_temporal_generalization import split_records_by_source_root  # noqa: E402
from scripts.train_c2c_v2_task_frame_v46_alignment import (  # noqa: E402
    TASK_FRAME_V46_RESIDUAL_SCALES,
    _build_arrays,
    _load_rows,
    _normalize_row_metadata,
)


def _score_residual_np(residual: np.ndarray) -> float:
    arr = np.asarray(residual, dtype=np.float32).reshape(-1)
    if arr.size < 4 or not np.all(np.isfinite(arr[:4])):
        return float("inf")
    return float(np.linalg.norm(arr[:3]) + abs(float(arr[3])))


def _score_residual_torch(residual: torch.Tensor) -> torch.Tensor:
    return torch.linalg.norm(residual[:, :3], dim=-1) + torch.abs(residual[:, 3])


def _score_command_np(pre: np.ndarray, post: np.ndarray, *, mode: str) -> float:
    if mode == "residual":
        return _score_residual_np(post)
    pre_arr = np.asarray(pre, dtype=np.float32).reshape(-1)
    post_arr = np.asarray(post, dtype=np.float32).reshape(-1)
    if pre_arr.size < 4 or post_arr.size < 4 or not np.all(np.isfinite(pre_arr[:4])) or not np.all(np.isfinite(post_arr[:4])):
        return float("inf")
    if mode == "outcome_utility":
        zero_score = _score_residual_np(pre_arr)
        post_score = _score_residual_np(post_arr)
        abs_pre = np.abs(pre_arr[:4])
        abs_post = np.abs(post_arr[:4])
        xy_contract = float(np.linalg.norm(post_arr[:2]) < np.linalg.norm(pre_arr[:2]) - 1.0e-7)
        z_contract = float(abs_post[2] < abs_pre[2] - 1.0e-7)
        yaw_contract = float(abs_post[3] < abs_pre[3] - 1.0e-7)
        combined_contract = float(post_score < zero_score - 1.0e-7)
        xy_worsen = float(np.linalg.norm(post_arr[:2]) > np.linalg.norm(pre_arr[:2]) + 1.0e-7)
        z_worsen = float(abs_post[2] > abs_pre[2] + 1.0e-7)
        yaw_worsen = float(abs_post[3] > abs_pre[3] + 1.0e-7)
        combined_worsen = float(post_score > zero_score + 1.0e-7)
        return float(
            -3.0 * combined_contract
            -1.0 * xy_contract
            -1.0 * z_contract
            -1.5 * yaw_contract
            +3.0 * combined_worsen
            +1.2 * xy_worsen
            +1.2 * z_worsen
            +2.0 * yaw_worsen
            +0.10 * _score_command_np(pre_arr, post_arr, mode="yaw_collateral")
        )
    if mode == "yaw_collateral":
        abs_pre = np.abs(pre_arr[:4])
        abs_post = np.abs(post_arr[:4])
        yaw_worsen = max(float(abs_post[3] - abs_pre[3]), 0.0) / 0.250
        xy_worsen = max(float(np.linalg.norm(post_arr[:2]) - np.linalg.norm(pre_arr[:2])), 0.0) / 0.040
        z_worsen = max(float(abs_post[2] - abs_pre[2]), 0.0) / 0.030
        return float((abs_post[3] / 0.250) + 8.0 * yaw_worsen + 3.0 * xy_worsen + 3.0 * z_worsen + 0.20 * np.linalg.norm(abs_post[:2] / 0.040))
    scales = np.asarray([0.040, 0.040, 0.030, 0.250], dtype=np.float32)
    abs_pre = np.abs(pre_arr[:4])
    abs_post = np.abs(post_arr[:4])
    normalized_post = abs_post / scales
    improved = abs_post < abs_pre
    worsen = np.maximum(abs_post - abs_pre, 0.0) / scales
    # The selector must not win by improving only one easy axis while pushing
    # another axis away from the handoff basin. Penalize missing per-axis
    # contraction and stronger axis worsen before residual magnitude.
    missing_penalty = np.where(improved, 0.0, 0.35).astype(np.float32)
    return float(np.sum(normalized_post) + 5.0 * np.sum(worsen) + np.sum(missing_penalty))


def _score_command_torch(pre: torch.Tensor, post: torch.Tensor, *, mode: str) -> torch.Tensor:
    if mode == "residual":
        return _score_residual_torch(post)
    if mode == "outcome_utility":
        pre_score = _score_residual_torch(pre)
        post_score = _score_residual_torch(post)
        abs_pre = torch.abs(pre[:, :4])
        abs_post = torch.abs(post[:, :4])
        xy_contract = (torch.linalg.norm(post[:, :2], dim=-1) < torch.linalg.norm(pre[:, :2], dim=-1) - 1.0e-7).to(dtype=post.dtype)
        z_contract = (abs_post[:, 2] < abs_pre[:, 2] - 1.0e-7).to(dtype=post.dtype)
        yaw_contract = (abs_post[:, 3] < abs_pre[:, 3] - 1.0e-7).to(dtype=post.dtype)
        combined_contract = (post_score < pre_score - 1.0e-7).to(dtype=post.dtype)
        xy_worsen = (torch.linalg.norm(post[:, :2], dim=-1) > torch.linalg.norm(pre[:, :2], dim=-1) + 1.0e-7).to(dtype=post.dtype)
        z_worsen = (abs_post[:, 2] > abs_pre[:, 2] + 1.0e-7).to(dtype=post.dtype)
        yaw_worsen = (abs_post[:, 3] > abs_pre[:, 3] + 1.0e-7).to(dtype=post.dtype)
        combined_worsen = (post_score > pre_score + 1.0e-7).to(dtype=post.dtype)
        return (
            -3.0 * combined_contract
            -1.0 * xy_contract
            -1.0 * z_contract
            -1.5 * yaw_contract
            +3.0 * combined_worsen
            +1.2 * xy_worsen
            +1.2 * z_worsen
            +2.0 * yaw_worsen
            +0.10 * _score_command_torch(pre, post, mode="yaw_collateral")
        )
    if mode == "yaw_collateral":
        abs_pre = torch.abs(pre[:, :4])
        abs_post = torch.abs(post[:, :4])
        yaw_worsen = torch.clamp(abs_post[:, 3] - abs_pre[:, 3], min=0.0) / 0.250
        xy_worsen = torch.clamp(torch.linalg.norm(post[:, :2], dim=-1) - torch.linalg.norm(pre[:, :2], dim=-1), min=0.0) / 0.040
        z_worsen = torch.clamp(abs_post[:, 2] - abs_pre[:, 2], min=0.0) / 0.030
        return (
            abs_post[:, 3] / 0.250
            + 8.0 * yaw_worsen
            + 3.0 * xy_worsen
            + 3.0 * z_worsen
            + 0.20 * torch.linalg.norm(abs_post[:, :2] / 0.040, dim=-1)
        )
    scales = torch.as_tensor([0.040, 0.040, 0.030, 0.250], dtype=post.dtype, device=post.device).reshape(1, -1)
    abs_pre = torch.abs(pre[:, :4])
    abs_post = torch.abs(post[:, :4])
    normalized_post = abs_post / scales
    worsen = torch.clamp(abs_post - abs_pre, min=0.0) / scales
    missing_penalty = (~(abs_post < abs_pre)).to(dtype=post.dtype) * 0.35
    return torch.sum(normalized_post, dim=-1) + 5.0 * torch.sum(worsen, dim=-1) + torch.sum(missing_penalty, dim=-1)


def _group_key(row: Mapping[str, Any]) -> str:
    source = str(row.get("source_eval_root", row.get("sequence_id", "")) or "")
    sequence = str(row.get("sequence_id", "") or "")
    prefix = sequence if sequence else source
    return f"{prefix}::ep{int(row.get('episode_idx', -1)):03d}:step{int(row.get('step_idx', row.get('step', -1))):04d}"


def _candidate_name(row: Mapping[str, Any]) -> str:
    return str(row.get("task_frame_v46_command_sweep_candidate_name", row.get("candidate_name", "unknown")) or "unknown")


def _candidate_command_feature_vector(
    row: Mapping[str, Any],
    command_6d: np.ndarray,
    *,
    mode: str = "raw6",
) -> np.ndarray:
    command = np.asarray(command_6d, dtype=np.float32).reshape(-1)
    if command.size < 6:
        command = np.pad(command, (0, 6 - command.size), mode="constant")
    command = command[:6].astype(np.float32)
    if str(mode) == "raw6":
        return command
    if str(mode) != "typed16":
        raise ValueError(f"unknown command feature mode: {mode}")
    name = _candidate_name(row).lower()
    zero = float(name == "zero" or np.linalg.norm(command) <= 1.0e-8)
    yaw = float("yaw" in name or abs(float(command[5])) > max(abs(float(command[0])), abs(float(command[1])), abs(float(command[2])), 1.0e-8))
    z = float("z_guard" in name or (abs(float(command[2])) > 1.0e-8 and yaw < 0.5))
    xy = float((abs(float(command[0])) > 1.0e-8 or abs(float(command[1])) > 1.0e-8) and yaw < 0.5 and z < 0.5)
    type_bits = np.asarray([zero, xy, z, yaw], dtype=np.float32)
    magnitudes = np.asarray(
        [
            np.linalg.norm(command[:2]) / 0.040,
            abs(float(command[2])) / 0.030,
            abs(float(command[5])) / 0.250,
        ],
        dtype=np.float32,
    )
    signs = np.asarray([np.sign(float(command[2])), np.sign(float(command[5]))], dtype=np.float32)
    command_norm = np.asarray([np.linalg.norm(command) / 0.250], dtype=np.float32)
    return np.concatenate([command, type_bits, magnitudes, signs, command_norm], axis=0).astype(np.float32)


def _zero_guard_margin_loss(
    pred_score: torch.Tensor,
    observed_score: torch.Tensor,
    *,
    zero_index: int | None,
    margin: float,
) -> torch.Tensor:
    """Keep the ranker from selecting commands that are worse than no-op.

    Scores are minimized.  Within one command-sweep window, offline transition
    labels decide which candidates beat the zero/no-op candidate.  Runtime never
    sees those labels; this only shapes the learned selector.
    """

    if zero_index is None or int(zero_index) < 0 or int(zero_index) >= int(pred_score.numel()):
        return pred_score.sum() * 0.0
    if pred_score.numel() <= 1:
        return pred_score.sum() * 0.0
    zidx = int(zero_index)
    device = pred_score.device
    mask = torch.ones_like(pred_score, dtype=torch.bool, device=device)
    mask[zidx] = False
    candidate_pred = pred_score[mask]
    candidate_observed = observed_score[mask]
    zero_pred = pred_score[zidx]
    zero_observed = observed_score[zidx]
    finite_mask = torch.isfinite(candidate_pred) & torch.isfinite(candidate_observed) & torch.isfinite(zero_pred) & torch.isfinite(zero_observed)
    if not bool(torch.any(finite_mask).item()):
        return pred_score.sum() * 0.0
    candidate_pred = candidate_pred[finite_mask]
    candidate_observed = candidate_observed[finite_mask]
    margin_value = torch.as_tensor(float(margin), dtype=pred_score.dtype, device=device)
    beats_zero = candidate_observed < (zero_observed - 1.0e-6)
    better_loss = torch.relu(candidate_pred + margin_value - zero_pred)[beats_zero]
    worse_loss = torch.relu(zero_pred + margin_value - candidate_pred)[~beats_zero]
    pieces = []
    if better_loss.numel() > 0:
        pieces.append(better_loss.mean())
    if worse_loss.numel() > 0:
        pieces.append(worse_loss.mean())
    if not pieces:
        return pred_score.sum() * 0.0
    return torch.stack(pieces).mean()


def _pairwise_oracle_margin_loss(
    pred_score: torch.Tensor,
    observed_score: torch.Tensor,
    *,
    zero_index: int | None,
    margin: float,
    zero_margin: float | None = None,
) -> torch.Tensor:
    """Rank the offline oracle ahead of non-oracle candidates in one window.

    Scores are minimized.  This complements cross-entropy by adding explicit
    pairwise margins for the held-out failure mode we care about: a command may
    look locally plausible but should not outrank the same-window oracle or
    zero/no-op when offline transition labels say it is worse.
    """

    if pred_score.numel() <= 1:
        return pred_score.sum() * 0.0
    finite = torch.isfinite(pred_score) & torch.isfinite(observed_score)
    if int(torch.sum(finite).item()) <= 1:
        return pred_score.sum() * 0.0

    pred = pred_score[finite]
    observed = observed_score[finite]
    oracle = int(torch.argmin(observed).item())
    device = pred_score.device
    margin_t = torch.as_tensor(float(margin), dtype=pred_score.dtype, device=device)
    candidate_mask = torch.ones_like(pred, dtype=torch.bool, device=device)
    candidate_mask[oracle] = False
    losses = [torch.relu(pred[oracle] + margin_t - pred[candidate_mask]).mean()]

    if zero_index is not None and 0 <= int(zero_index) < int(pred_score.numel()):
        original_to_compact = torch.full((int(pred_score.numel()),), -1, dtype=torch.long, device=device)
        original_to_compact[torch.nonzero(finite, as_tuple=False).reshape(-1)] = torch.arange(
            int(torch.sum(finite).item()),
            dtype=torch.long,
            device=device,
        )
        compact_zero = int(original_to_compact[int(zero_index)].item())
        if compact_zero >= 0:
            zero_margin_t = torch.as_tensor(float(margin if zero_margin is None else zero_margin), dtype=pred_score.dtype, device=device)
            zero_observed = observed[compact_zero]
            worse_than_zero = observed >= zero_observed - 1.0e-6
            worse_than_zero[compact_zero] = False
            if bool(torch.any(worse_than_zero).item()):
                losses.append(torch.relu(pred[compact_zero] + zero_margin_t - pred[worse_than_zero]).mean())

    return torch.stack(losses).mean()


def _support_target_from_zero_or_pre(
    observed_score: torch.Tensor,
    residual: torch.Tensor,
    *,
    zero_index: int | None,
) -> torch.Tensor:
    if zero_index is not None and 0 <= int(zero_index) < int(observed_score.numel()):
        baseline = observed_score[int(zero_index)]
    else:
        baseline = _score_residual_torch(residual)
    return (observed_score < baseline - 1.0e-6).float()


def _adjust_rank_score_with_support(
    score: torch.Tensor,
    support: torch.Tensor,
    *,
    support_score_penalty: float,
) -> torch.Tensor:
    if float(support_score_penalty) <= 0.0:
        return score
    return score + float(support_score_penalty) * torch.clamp(1.0 - support, min=0.0, max=1.0)


def _command_outcome_targets(
    residual: torch.Tensor,
    next_residual: torch.Tensor,
    observed_score: torch.Tensor,
    *,
    zero_index: int | None,
) -> torch.Tensor:
    support = _support_target_from_zero_or_pre(observed_score, residual, zero_index=zero_index)
    pre_score = _score_residual_torch(residual)
    post_score = _score_residual_torch(next_residual)
    abs_pre = torch.abs(residual[:, :4])
    abs_post = torch.abs(next_residual[:, :4])
    xy_pre = torch.linalg.norm(residual[:, :2], dim=-1)
    xy_post = torch.linalg.norm(next_residual[:, :2], dim=-1)
    targets = torch.stack(
        [
            support,
            (xy_post < xy_pre - 1.0e-7).to(dtype=residual.dtype),
            (abs_post[:, 2] < abs_pre[:, 2] - 1.0e-7).to(dtype=residual.dtype),
            (abs_post[:, 3] < abs_pre[:, 3] - 1.0e-7).to(dtype=residual.dtype),
            (post_score < pre_score - 1.0e-7).to(dtype=residual.dtype),
            (xy_post > xy_pre + 1.0e-7).to(dtype=residual.dtype),
            (abs_post[:, 2] > abs_pre[:, 2] + 1.0e-7).to(dtype=residual.dtype),
            (abs_post[:, 3] > abs_pre[:, 3] + 1.0e-7).to(dtype=residual.dtype),
            (post_score > pre_score + 1.0e-7).to(dtype=residual.dtype),
        ],
        dim=-1,
    )
    return targets


def _predicted_outcome_utility(logits: torch.Tensor) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    beat_zero = probs[:, 0]
    xy_contract = probs[:, 1]
    z_contract = probs[:, 2]
    yaw_contract = probs[:, 3]
    combined_contract = probs[:, 4]
    xy_worsen = probs[:, 5]
    z_worsen = probs[:, 6]
    yaw_worsen = probs[:, 7]
    combined_worsen = probs[:, 8]
    return (
        -3.0 * beat_zero
        -0.8 * combined_contract
        -0.5 * xy_contract
        -0.5 * z_contract
        -0.8 * yaw_contract
        +3.0 * combined_worsen
        +1.0 * xy_worsen
        +1.0 * z_worsen
        +1.6 * yaw_worsen
    )


def _parse_source_eval_root_args(values: list[str] | None) -> set[str]:
    roots: set[str] = set()
    for value in values or []:
        for part in str(value).split(","):
            root = part.strip()
            if root:
                roots.add(root)
    return roots


def _split_groups(group_keys: list[str], *, val_fraction: float, seed: int) -> tuple[set[str], set[str]]:
    keys = sorted(set(group_keys))
    rng = np.random.default_rng(int(seed))
    order = list(keys)
    rng.shuffle(order)
    val_count = max(1, int(round(len(order) * float(val_fraction)))) if len(order) > 1 else 0
    val = set(order[:val_count])
    train = set(order[val_count:])
    if not train and val:
        moved = sorted(val)[0]
        val.remove(moved)
        train.add(moved)
    return train, val


def _split_ranker_rows(
    rows: list[dict[str, Any]],
    *,
    split_mode: str,
    val_fraction: float,
    seed: int,
    test_fraction: float = 0.0,
    train_source_eval_roots: set[str] | None = None,
    val_source_eval_roots: set[str] | None = None,
    test_source_eval_roots: set[str] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    split = split_records_by_source_root(
        rows,
        split_mode=str(split_mode),
        val_fraction=float(val_fraction),
        test_fraction=float(test_fraction),
        seed=int(seed),
        train_source_eval_roots=train_source_eval_roots,
        val_source_eval_roots=val_source_eval_roots,
        test_source_eval_roots=test_source_eval_roots,
    )
    return (
        [dict(row) for row in split.train_records],
        [dict(row) for row in split.val_records],
        {
            "split_mode": split.split_mode,
            "test_fraction": float(test_fraction),
            "train_source_eval_roots": list(split.train_source_eval_roots),
            "val_source_eval_roots": list(split.val_source_eval_roots),
            "test_source_eval_roots": list(split.test_source_eval_roots),
            "requested_train_source_eval_roots": sorted(train_source_eval_roots or set()),
            "requested_val_source_eval_roots": sorted(val_source_eval_roots or set()),
            "requested_test_source_eval_roots": sorted(test_source_eval_roots or set()),
        },
    )


def _indices_by_group(kept: list[dict[str, Any]]) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for idx, row in enumerate(kept):
        groups[_group_key(row)].append(idx)
    return dict(groups)


def _slice_arrays(arrays: dict[str, np.ndarray], indices: list[int]) -> dict[str, np.ndarray]:
    idx = np.asarray(indices, dtype=np.int64)
    return {key: value[idx] for key, value in arrays.items()}


def _build_group_training_arrays(
    rows: list[dict[str, Any]],
    *,
    command_feature_mode: str,
    image_crop_size: int,
    image_resize_size: int,
    history_window_size: int,
    max_abs_xy_label: float,
    max_abs_z_label: float,
    max_abs_yaw_label: float,
    near_field_xy_radius: float,
    near_field_z_radius: float,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]], dict[str, list[int]]]:
    arrays, kept = _build_arrays(
        rows,
        image_crop_size=image_crop_size,
        image_resize_size=image_resize_size,
        history_window_size=history_window_size,
        proprio_dim=15,
        planner_prior_dim=6,
        max_abs_xy_label=max_abs_xy_label,
        max_abs_z_label=max_abs_z_label,
        max_abs_yaw_label=max_abs_yaw_label,
        near_field_xy_radius=near_field_xy_radius,
        near_field_z_radius=near_field_z_radius,
    )
    command_features = [
        _candidate_command_feature_vector(row, command, mode=command_feature_mode)
        for row, command in zip(kept, arrays.get("command_6d", np.zeros((len(kept), 6), dtype=np.float32)))
    ]
    arrays["command_features"] = np.asarray(command_features, dtype=np.float32)
    groups = {key: idxs for key, idxs in _indices_by_group(kept).items() if len(idxs) >= 2}
    if not groups:
        raise RuntimeError("candidate ranker needs at least one episode+step group with two or more candidates")
    return arrays, kept, groups


def _group_metrics(
    model: TaskFrameV46AlignmentNet,
    arrays: dict[str, np.ndarray],
    kept: list[dict[str, Any]],
    groups: dict[str, list[int]],
    *,
    device: str,
    rank_score_mode: str,
    support_score_penalty: float = 0.0,
) -> dict[str, Any]:
    model.eval()
    metrics: dict[str, Any] = {
        "groups": int(len(groups)),
        "rows": int(sum(len(v) for v in groups.values())),
        "top1_best_score_match": 0.0,
        "top1_xy_contraction": 0.0,
        "top1_z_contraction": 0.0,
        "top1_yaw_contraction": 0.0,
        "top1_combined_contraction": 0.0,
        "oracle_xy_contraction": 0.0,
        "oracle_z_contraction": 0.0,
        "oracle_yaw_contraction": 0.0,
        "oracle_combined_contraction": 0.0,
        "zero_xy_contraction": 0.0,
        "zero_z_contraction": 0.0,
        "zero_yaw_contraction": 0.0,
        "zero_combined_contraction": 0.0,
        "selected_candidate_counts": {},
        "oracle_candidate_counts": {},
        "group_details": {},
    }
    selected_counts: defaultdict[str, int] = defaultdict(int)
    oracle_counts: defaultdict[str, int] = defaultdict(int)
    rows_top: list[dict[str, Any]] = []
    with torch.no_grad():
        for key, idxs in sorted(groups.items()):
            subset = _slice_arrays(arrays, idxs)
            out = model(
                torch.as_tensor(subset["image"], dtype=torch.float32, device=device),
                torch.as_tensor(subset["scalar"], dtype=torch.float32, device=device),
                torch.as_tensor(subset["history"], dtype=torch.float32, device=device),
                torch.as_tensor(subset["proprio"], dtype=torch.float32, device=device),
                torch.as_tensor(subset["planner"], dtype=torch.float32, device=device),
                torch.as_tensor(subset.get("command_features", subset["command_6d"]), dtype=torch.float32, device=device),
            )
            residual = torch.as_tensor(subset["residual"], dtype=torch.float32, device=device)
            pred_post = residual + out["command_delta"]
            if rank_score_mode == "outcome_utility":
                raw_pred_score_t = _predicted_outcome_utility(out["command_outcome_logits"])
            else:
                raw_pred_score_t = _score_command_torch(residual, pred_post, mode=rank_score_mode)
            pred_score_t = _adjust_rank_score_with_support(
                raw_pred_score_t,
                out["command_support"],
                support_score_penalty=float(support_score_penalty),
            )
            pred_score = pred_score_t.detach().cpu().numpy()
            pre = subset["residual"]
            post = subset["next_residual"]
            observed_scores = np.asarray(
                [_score_command_np(a, b, mode=rank_score_mode) for a, b in zip(pre, post)],
                dtype=np.float32,
            )
            oracle_local = int(np.argmin(observed_scores))
            selected_local = int(np.argmin(pred_score))
            zero_local = next((i for i, idx in enumerate(idxs) if _candidate_name(kept[idx]) == "zero"), oracle_local)
            selected_global = idxs[selected_local]
            oracle_global = idxs[oracle_local]
            zero_global = idxs[zero_local]
            selected_counts[_candidate_name(kept[selected_global])] += 1
            oracle_counts[_candidate_name(kept[oracle_global])] += 1

            def flags(local_idx: int) -> tuple[bool, bool, bool, bool]:
                a = pre[local_idx]
                b = post[local_idx]
                xy = bool(np.linalg.norm(b[:2]) < np.linalg.norm(a[:2]))
                z = bool(abs(float(b[2])) < abs(float(a[2])))
                yaw = bool(abs(float(b[3])) < abs(float(a[3])))
                combined = bool(_score_residual_np(b) < _score_residual_np(a))
                return xy, z, yaw, combined

            selected_flags = flags(selected_local)
            oracle_flags = flags(oracle_local)
            zero_flags = flags(zero_local)
            rows_top.append(
                {
                    "group": key,
                    "selected_candidate": _candidate_name(kept[selected_global]),
                    "oracle_candidate": _candidate_name(kept[oracle_global]),
                    "zero_available": bool(_candidate_name(kept[zero_global]) == "zero"),
                    "selected_score_pred": float(pred_score[selected_local]),
                    "selected_support_pred": float(out["command_support"].detach().cpu().numpy()[selected_local]),
                    "selected_score_observed": float(observed_scores[selected_local]),
                    "oracle_score_observed": float(observed_scores[oracle_local]),
                    "selected_flags": selected_flags,
                    "oracle_flags": oracle_flags,
                    "zero_flags": zero_flags,
                }
            )
    n = max(1, len(rows_top))
    metrics["top1_best_score_match"] = float(sum(row["selected_candidate"] == row["oracle_candidate"] for row in rows_top) / n)
    for prefix, field in (("top1", "selected_flags"), ("oracle", "oracle_flags"), ("zero", "zero_flags")):
        metrics[f"{prefix}_xy_contraction"] = float(sum(bool(row[field][0]) for row in rows_top) / n)
        metrics[f"{prefix}_z_contraction"] = float(sum(bool(row[field][1]) for row in rows_top) / n)
        metrics[f"{prefix}_yaw_contraction"] = float(sum(bool(row[field][2]) for row in rows_top) / n)
        metrics[f"{prefix}_combined_contraction"] = float(sum(bool(row[field][3]) for row in rows_top) / n)
    metrics["selected_candidate_counts"] = dict(sorted(selected_counts.items()))
    metrics["oracle_candidate_counts"] = dict(sorted(oracle_counts.items()))
    metrics["group_details"] = {row["group"]: row for row in rows_top}
    return metrics


def train(
    dataset_jsonl: list[Path],
    *,
    output_checkpoint: Path,
    output_json: Path,
    val_fraction: float = 0.2,
    test_fraction: float = 0.0,
    split_mode: str = "root",
    epochs: int = 80,
    lr: float = 5.0e-4,
    seed: int = 7,
    image_hidden_dim: int = 128,
    fusion_hidden_dim: int = 128,
    history_window_size: int = RUNTIME_XY_SPATIAL_TEMPORAL_HISTORY_WINDOW,
    image_crop_size: int = RUNTIME_XY_SPATIAL_TEMPORAL_CROP_SIZE,
    image_resize_size: int = RUNTIME_XY_SPATIAL_TEMPORAL_IMAGE_SIZE,
    max_abs_xy_label: float = 0.090,
    max_abs_z_label: float = 0.080,
    max_abs_yaw_label: float = 0.350,
    near_field_xy_radius: float = 0.060,
    near_field_z_radius: float = 0.040,
    ranking_temperature: float = 0.010,
    rank_score_mode: str = "residual",
    zero_guard_margin: float = 0.050,
    zero_guard_weight: float = 1.000,
    support_score_penalty: float = 0.250,
    outcome_loss_weight: float = 1.000,
    pairwise_margin: float = 0.050,
    pairwise_weight: float = 0.000,
    pairwise_zero_margin: float | None = None,
    command_feature_mode: str = "raw6",
    train_source_eval_roots: set[str] | None = None,
    val_source_eval_roots: set[str] | None = None,
    test_source_eval_roots: set[str] | None = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> dict[str, Any]:
    torch.manual_seed(int(seed))
    rows = [_normalize_row_metadata(row) for row in _load_rows(dataset_jsonl)]
    train_rows, val_rows, split_meta = _split_ranker_rows(
        rows,
        split_mode=split_mode,
        val_fraction=val_fraction,
        test_fraction=test_fraction,
        seed=seed,
        train_source_eval_roots=train_source_eval_roots,
        val_source_eval_roots=val_source_eval_roots,
        test_source_eval_roots=test_source_eval_roots,
    )
    if not val_rows:
        raise RuntimeError("candidate ranker validation split is empty")
    train_arrays, train_kept, train_group_indices = _build_group_training_arrays(
        train_rows,
        command_feature_mode=str(command_feature_mode),
        image_crop_size=image_crop_size,
        image_resize_size=image_resize_size,
        history_window_size=history_window_size,
        max_abs_xy_label=max_abs_xy_label,
        max_abs_z_label=max_abs_z_label,
        max_abs_yaw_label=max_abs_yaw_label,
        near_field_xy_radius=near_field_xy_radius,
        near_field_z_radius=near_field_z_radius,
    )
    val_arrays, val_kept, val_group_indices = _build_group_training_arrays(
        val_rows,
        command_feature_mode=str(command_feature_mode),
        image_crop_size=image_crop_size,
        image_resize_size=image_resize_size,
        history_window_size=history_window_size,
        max_abs_xy_label=max_abs_xy_label,
        max_abs_z_label=max_abs_z_label,
        max_abs_yaw_label=max_abs_yaw_label,
        near_field_xy_radius=near_field_xy_radius,
        near_field_z_radius=near_field_z_radius,
    )
    model = TaskFrameV46AlignmentNet(
        image_hidden_dim=image_hidden_dim,
        scalar_feature_dim=len(TASK_FRAME_READINESS_FEATURE_NAMES),
        history_feature_dim=max(1, train_arrays["history"].shape[1] // max(1, history_window_size)),
        history_window_size=history_window_size,
        command_feature_dim=int(train_arrays.get("command_features", train_arrays["command_6d"]).shape[1]),
        fusion_hidden_dim=fusion_hidden_dim,
        risk_classes=TASK_FRAME_V46_RISK_CLASSES,
        max_abs_xy=max_abs_xy_label,
        max_abs_z=max_abs_z_label,
        max_abs_yaw=max_abs_yaw_label,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=1.0e-4)
    scales = TASK_FRAME_V46_RESIDUAL_SCALES.to(device=device).reshape(1, -1)
    group_items = list(train_group_indices.items())
    rng = np.random.default_rng(int(seed))
    for _epoch in range(int(epochs)):
        model.train()
        rng.shuffle(group_items)
        for _key, idxs in group_items:
            subset = _slice_arrays(train_arrays, idxs)
            residual = torch.as_tensor(subset["residual"], dtype=torch.float32, device=device)
            next_residual = torch.as_tensor(subset["next_residual"], dtype=torch.float32, device=device)
            out = model(
                torch.as_tensor(subset["image"], dtype=torch.float32, device=device),
                torch.as_tensor(subset["scalar"], dtype=torch.float32, device=device),
                torch.as_tensor(subset["history"], dtype=torch.float32, device=device),
                torch.as_tensor(subset["proprio"], dtype=torch.float32, device=device),
                torch.as_tensor(subset["planner"], dtype=torch.float32, device=device),
                torch.as_tensor(subset.get("command_features", subset["command_6d"]), dtype=torch.float32, device=device),
            )
            observed_delta = next_residual - residual
            pred_delta = out["command_delta"]
            pred_post = residual + pred_delta
            observed_score = _score_command_torch(residual, next_residual, mode=rank_score_mode)
            target_idx = torch.argmin(observed_score).reshape(1)
            zero_local = next((i for i, idx in enumerate(idxs) if _candidate_name(train_kept[idx]) == "zero"), None)
            if rank_score_mode == "outcome_utility":
                pred_score = _predicted_outcome_utility(out["command_outcome_logits"])
            else:
                pred_score = _score_command_torch(residual, pred_post, mode=rank_score_mode)
            adjusted_pred_score = _adjust_rank_score_with_support(
                pred_score,
                out["command_support"],
                support_score_penalty=float(support_score_penalty),
            )
            ranking_loss = F.cross_entropy((-adjusted_pred_score / float(ranking_temperature)).reshape(1, -1), target_idx)
            zero_guard_loss = _zero_guard_margin_loss(
                adjusted_pred_score,
                observed_score,
                zero_index=zero_local,
                margin=float(zero_guard_margin),
            )
            pairwise_loss = _pairwise_oracle_margin_loss(
                adjusted_pred_score,
                observed_score,
                zero_index=zero_local,
                margin=float(pairwise_margin),
                zero_margin=pairwise_zero_margin,
            )
            delta_loss = F.smooth_l1_loss(pred_delta / scales, observed_delta / scales)
            support_target = _support_target_from_zero_or_pre(observed_score, residual, zero_index=zero_local)
            support_loss = F.binary_cross_entropy(out["command_support"], support_target)
            outcome_target = _command_outcome_targets(residual, next_residual, observed_score, zero_index=zero_local)
            outcome_loss = F.binary_cross_entropy_with_logits(out["command_outcome_logits"], outcome_target)
            post_loss = F.smooth_l1_loss(pred_post / scales, next_residual / scales)
            opt.zero_grad(set_to_none=True)
            (
                2.5 * ranking_loss
                + float(zero_guard_weight) * zero_guard_loss
                + float(pairwise_weight) * pairwise_loss
                + 0.8 * delta_loss
                + 0.6 * support_loss
                + float(outcome_loss_weight) * outcome_loss
                + 0.5 * post_loss
            ).backward()
            opt.step()
    train_metrics = _group_metrics(
        model,
        train_arrays,
        train_kept,
        train_group_indices,
        device=device,
        rank_score_mode=rank_score_mode,
        support_score_penalty=float(support_score_penalty),
    )
    val_metrics = _group_metrics(
        model,
        val_arrays,
        val_kept,
        val_group_indices,
        device=device,
        rank_score_mode=rank_score_mode,
        support_score_penalty=float(support_score_penalty),
    )
    metadata = {
        "schema_version": "c2c_v2_task_frame_v46_candidate_ranker_report_v1",
        "model": "v46_unified_task_frame_alignment_candidate_ranker",
        "dataset_jsonl": [str(path) for path in dataset_jsonl],
        "split_mode": str(split_mode),
        "val_fraction": float(val_fraction),
        "test_fraction": float(test_fraction),
        "train_groups": sorted(train_group_indices),
        "val_groups": sorted(val_group_indices),
        "train_source_eval_roots": split_meta["train_source_eval_roots"],
        "val_source_eval_roots": split_meta["val_source_eval_roots"],
        "test_source_eval_roots": split_meta["test_source_eval_roots"],
        "requested_train_source_eval_roots": split_meta["requested_train_source_eval_roots"],
        "requested_val_source_eval_roots": split_meta["requested_val_source_eval_roots"],
        "requested_test_source_eval_roots": split_meta["requested_test_source_eval_roots"],
        "train_rows": int(train_arrays["image"].shape[0]),
        "val_rows": int(val_arrays["image"].shape[0]),
        "max_abs_xy_label": float(max_abs_xy_label),
        "max_abs_z_label": float(max_abs_z_label),
        "max_abs_yaw_label": float(max_abs_yaw_label),
        "ranking_temperature": float(ranking_temperature),
        "rank_score_mode": str(rank_score_mode),
        "zero_guard_margin": float(zero_guard_margin),
        "zero_guard_weight": float(zero_guard_weight),
        "support_score_penalty": float(support_score_penalty),
        "outcome_loss_weight": float(outcome_loss_weight),
        "pairwise_margin": float(pairwise_margin),
        "pairwise_weight": float(pairwise_weight),
        "pairwise_zero_margin": None if pairwise_zero_margin is None else float(pairwise_zero_margin),
        "command_feature_mode": str(command_feature_mode),
        "command_feature_dim": int(train_arrays.get("command_features", train_arrays["command_6d"]).shape[1]),
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "uses_privileged_runtime": False,
        "uses_privileged_label_for_training": True,
        "privileged_label_boundary": "offline_pre_post_transition_labels_only",
        "upgrade_gate": "pending_large_random_holdout_and_closed_loop_insert_success",
    }
    output_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    save_task_frame_v46_alignment_checkpoint(
        output_checkpoint,
        model,
        metadata=metadata,
        image_crop_size=image_crop_size,
        image_resize_size=image_resize_size,
    )
    output_json.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_jsonl", nargs="+", type=Path, required=True)
    parser.add_argument("--output_checkpoint", type=Path, required=True)
    parser.add_argument("--output_json", type=Path, required=True)
    parser.add_argument("--val_fraction", type=float, default=0.2)
    parser.add_argument("--test_fraction", type=float, default=0.0)
    parser.add_argument("--split_mode", type=str, default="root", choices=("root", "episode"))
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=5.0e-4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--image_hidden_dim", type=int, default=128)
    parser.add_argument("--fusion_hidden_dim", type=int, default=128)
    parser.add_argument("--history_window_size", type=int, default=RUNTIME_XY_SPATIAL_TEMPORAL_HISTORY_WINDOW)
    parser.add_argument("--image_crop_size", type=int, default=RUNTIME_XY_SPATIAL_TEMPORAL_CROP_SIZE)
    parser.add_argument("--image_resize_size", type=int, default=RUNTIME_XY_SPATIAL_TEMPORAL_IMAGE_SIZE)
    parser.add_argument("--max_abs_xy_label", type=float, default=0.090)
    parser.add_argument("--max_abs_z_label", type=float, default=0.080)
    parser.add_argument("--max_abs_yaw_label", type=float, default=0.350)
    parser.add_argument("--near_field_xy_radius", type=float, default=0.060)
    parser.add_argument("--near_field_z_radius", type=float, default=0.040)
    parser.add_argument("--ranking_temperature", type=float, default=0.010)
    parser.add_argument("--rank_score_mode", type=str, default="residual", choices=("residual", "axis_balanced", "yaw_collateral", "outcome_utility"))
    parser.add_argument("--zero_guard_margin", type=float, default=0.050)
    parser.add_argument("--zero_guard_weight", type=float, default=1.000)
    parser.add_argument("--support_score_penalty", type=float, default=0.250)
    parser.add_argument("--outcome_loss_weight", type=float, default=1.000)
    parser.add_argument("--pairwise_margin", type=float, default=0.050)
    parser.add_argument("--pairwise_weight", type=float, default=0.000)
    parser.add_argument("--pairwise_zero_margin", type=float, default=-1.0)
    parser.add_argument("--command_feature_mode", type=str, default="raw6", choices=("raw6", "typed16"))
    parser.add_argument(
        "--train_source_eval_root",
        action="append",
        default=[],
        help="Source eval root(s) forced into train; repeat or comma-separate.",
    )
    parser.add_argument(
        "--val_source_eval_root",
        action="append",
        default=[],
        help="Source eval root(s) forced into validation; repeat or comma-separate.",
    )
    parser.add_argument(
        "--test_source_eval_root",
        action="append",
        default=[],
        help="Source eval root(s) forced into test/unused holdout; repeat or comma-separate.",
    )
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = train(
        list(args.dataset_jsonl),
        output_checkpoint=args.output_checkpoint,
        output_json=args.output_json,
        val_fraction=float(args.val_fraction),
        test_fraction=float(args.test_fraction),
        split_mode=str(args.split_mode),
        epochs=int(args.epochs),
        lr=float(args.lr),
        seed=int(args.seed),
        image_hidden_dim=int(args.image_hidden_dim),
        fusion_hidden_dim=int(args.fusion_hidden_dim),
        history_window_size=int(args.history_window_size),
        image_crop_size=int(args.image_crop_size),
        image_resize_size=int(args.image_resize_size),
        max_abs_xy_label=float(args.max_abs_xy_label),
        max_abs_z_label=float(args.max_abs_z_label),
        max_abs_yaw_label=float(args.max_abs_yaw_label),
        near_field_xy_radius=float(args.near_field_xy_radius),
        near_field_z_radius=float(args.near_field_z_radius),
        ranking_temperature=float(args.ranking_temperature),
        rank_score_mode=str(args.rank_score_mode),
        zero_guard_margin=float(args.zero_guard_margin),
        zero_guard_weight=float(args.zero_guard_weight),
        support_score_penalty=float(args.support_score_penalty),
        outcome_loss_weight=float(args.outcome_loss_weight),
        pairwise_margin=float(args.pairwise_margin),
        pairwise_weight=float(args.pairwise_weight),
        pairwise_zero_margin=None if float(args.pairwise_zero_margin) < 0.0 else float(args.pairwise_zero_margin),
        command_feature_mode=str(args.command_feature_mode),
        train_source_eval_roots=_parse_source_eval_root_args(args.train_source_eval_root),
        val_source_eval_roots=_parse_source_eval_root_args(args.val_source_eval_root),
        test_source_eval_roots=_parse_source_eval_root_args(args.test_source_eval_root),
        device=str(args.device),
    )
    print(json.dumps({"train_metrics": report["train_metrics"], "val_metrics": report["val_metrics"], "upgrade_gate": report["upgrade_gate"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
