#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset

from prismatic.models.student_candidate_evaluator_v2 import StudentCandidateEvaluatorV2
from prismatic.models.student_handoff_state_head_v2 import StudentHandoffStateHeadV2


class CandidateRankingDataset(Dataset):
    def __init__(self, npz_paths: list[str]):
        chunks = []
        for npz_path in npz_paths:
            raw = np.load(npz_path, allow_pickle=False)
            chunks.append({k: np.asarray(raw[k]) for k in raw.files})
        keys = sorted(set().union(*(c.keys() for c in chunks)))
        data: dict[str, np.ndarray] = {}
        special_index_defaults = {
            "candidate_best_index": -1,
            "candidate_baseline_index": -1,
            "candidate_bad_index": -1,
            "oracle_candidate_index": -1,
            "pred_candidate_index": -1,
            "runtime_selected_candidate_index": -1,
        }
        candidate_tensor_keys = {
            "candidate_actions_local",
            "candidate_mask",
            "candidate_oracle_score",
        }

        max_candidates = 0
        for c in chunks:
            if "candidate_actions_local" in c:
                max_candidates = max(max_candidates, int(np.asarray(c["candidate_actions_local"]).shape[1]))
            elif "candidate_mask" in c:
                max_candidates = max(max_candidates, int(np.asarray(c["candidate_mask"]).shape[1]))
            elif "candidate_oracle_score" in c:
                max_candidates = max(max_candidates, int(np.asarray(c["candidate_oracle_score"]).shape[1]))

        def pad_candidate_tensor(key: str, arr: np.ndarray | None, n: int, dtype: np.dtype) -> np.ndarray:
            if key == "candidate_actions_local":
                out = np.zeros((n, max_candidates, 6), dtype=dtype)
            else:
                fill = -1e9 if key == "candidate_oracle_score" else 0.0
                out = np.full((n, max_candidates), fill, dtype=dtype)
            if arr is None:
                return out
            width = int(arr.shape[1])
            out[:, :width, ...] = arr.astype(dtype, copy=False)
            return out

        for key in keys:
            exemplar = next((c[key] for c in chunks if key in c), None)
            if exemplar is None:
                continue
            arrs = []
            for c in chunks:
                n = int(next(iter(c.values())).shape[0])
                arr = np.asarray(c[key]) if key in c else None
                if key in candidate_tensor_keys:
                    arrs.append(pad_candidate_tensor(key, arr, n, exemplar.dtype))
                    continue
                if arr is not None and tuple(arr.shape[1:]) == tuple(exemplar.shape[1:]):
                    arrs.append(arr)
                    continue
                shape = (n,) + tuple(exemplar.shape[1:])
                if key in special_index_defaults:
                    arrs.append(np.full(shape, special_index_defaults[key], dtype=exemplar.dtype))
                elif exemplar.dtype.kind in ("U", "S", "O"):
                    arrs.append(np.full(shape, "", dtype=exemplar.dtype))
                else:
                    arrs.append(np.zeros(shape, dtype=exemplar.dtype))
            data[key] = np.concatenate(arrs, axis=0)

        if "candidate_mask" in data:
            mask = np.asarray(data["candidate_mask"], dtype=np.float32) > 0.5
            width = mask.shape[1]
            for idx_key in ("candidate_best_index", "candidate_baseline_index", "candidate_bad_index"):
                if idx_key not in data:
                    continue
                idx = np.asarray(data[idx_key], dtype=np.int64)
                invalid = (idx < 0) | (idx >= width)
                if np.any(invalid):
                    idx = idx.copy()
                    idx[invalid] = -1
                    data[idx_key] = idx
                if idx_key == "candidate_best_index":
                    # best index must always be valid for CE; fall back to best valid oracle score.
                    bad = (idx < 0) | (idx >= width)
                    if np.any(bad):
                        scores = np.asarray(data["candidate_oracle_score"], dtype=np.float32)
                        valid_scores = np.where(mask, scores, -1e9)
                        repaired = np.argmax(valid_scores, axis=1).astype(np.int64)
                        idx = np.where(bad, repaired, idx)
                        data[idx_key] = idx
                elif idx_key == "candidate_baseline_index":
                    bad = (idx < 0) | (idx >= width)
                    if np.any(bad):
                        fallback = np.argmax(mask.astype(np.int64), axis=1).astype(np.int64)
                        idx = np.where(bad, fallback, idx)
                        data[idx_key] = idx
        self.data = data
        self.length = int(self.data["candidate_best_index"].shape[0])

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        if "handoff_latent" in self.data:
            out = {
                "handoff_latent": torch.from_numpy(self.data["handoff_latent"][idx].astype(np.float32)),
                "proxy_current_delta_basin_target": torch.from_numpy(self.data["proxy_current_delta_basin_target"][idx].astype(np.float32)),
                "candidate_actions_local": torch.from_numpy(self.data["candidate_actions_local"][idx].astype(np.float32)),
                "candidate_mask": torch.from_numpy(self.data["candidate_mask"][idx].astype(np.float32)),
                "candidate_oracle_score": torch.from_numpy(self.data["candidate_oracle_score"][idx].astype(np.float32)),
                "candidate_best_index": torch.tensor(int(self.data["candidate_best_index"][idx]), dtype=torch.long),
                "candidate_baseline_index": torch.tensor(int(self.data["candidate_baseline_index"][idx]), dtype=torch.long),
                "candidate_bad_index": torch.tensor(int(self.data.get("candidate_bad_index", np.full((self.length,), -1, dtype=np.int64))[idx]), dtype=torch.long),
                "sample_weight": torch.tensor(float(self.data.get("sample_weight", np.ones((self.length,), dtype=np.float32))[idx]), dtype=torch.float32),
                "is_shadow_hard_negative": torch.tensor(float(self.data.get("is_shadow_hard_negative", np.zeros((self.length,), dtype=np.float32))[idx]), dtype=torch.float32),
                "pred_worse_than_baseline": torch.tensor(float(self.data.get("pred_worse_than_baseline", np.zeros((self.length,), dtype=np.float32))[idx]), dtype=torch.float32),
                "pred_better_than_baseline": torch.tensor(float(self.data.get("pred_better_than_baseline", np.zeros((self.length,), dtype=np.float32))[idx]), dtype=torch.float32),
                "pred_has_yaw": torch.tensor(float(self.data.get("pred_has_yaw", np.zeros((self.length,), dtype=np.float32))[idx]), dtype=torch.float32),
                "oracle_has_yaw": torch.tensor(float(self.data.get("oracle_has_yaw", np.zeros((self.length,), dtype=np.float32))[idx]), dtype=torch.float32),
                "baseline_has_yaw": torch.tensor(float(self.data.get("baseline_has_yaw", np.zeros((self.length,), dtype=np.float32))[idx]), dtype=torch.float32),
                "pred_large_yaw_negative": torch.tensor(float(self.data.get("pred_large_yaw_negative", np.zeros((self.length,), dtype=np.float32))[idx]), dtype=torch.float32),
                "pred_large_yaw_positive": torch.tensor(float(self.data.get("pred_large_yaw_positive", np.zeros((self.length,), dtype=np.float32))[idx]), dtype=torch.float32),
                "pred_small_yaw_positive": torch.tensor(float(self.data.get("pred_small_yaw_positive", np.zeros((self.length,), dtype=np.float32))[idx]), dtype=torch.float32),
                "pred_yaw_bucket": torch.tensor(int(self.data.get("pred_yaw_bucket", np.zeros((self.length,), dtype=np.int64))[idx]), dtype=torch.long),
                "baseline_yaw_bucket": torch.tensor(int(self.data.get("baseline_yaw_bucket", np.zeros((self.length,), dtype=np.int64))[idx]), dtype=torch.long),
                "oracle_yaw_bucket": torch.tensor(int(self.data.get("oracle_yaw_bucket", np.zeros((self.length,), dtype=np.int64))[idx]), dtype=torch.long),
                "best_yaw_bucket": torch.tensor(int(self.data.get("best_yaw_bucket", np.zeros((self.length,), dtype=np.int64))[idx]), dtype=torch.long),
                "hard_episode_negative": torch.tensor(float(self.data.get("hard_episode_negative", np.zeros((self.length,), dtype=np.float32))[idx]), dtype=torch.float32),
                "mode_target_override": torch.tensor(int(self.data.get("mode_target_override", np.full((self.length,), -1, dtype=np.int64))[idx]), dtype=torch.long),
                "shadow_mode": torch.tensor(int(self.data.get("shadow_mode", np.full((self.length,), -1, dtype=np.int64))[idx]), dtype=torch.long),
                "hardneg_mode_keep_failure": torch.tensor(float(self.data.get("hardneg_mode_keep_failure", np.zeros((self.length,), dtype=np.float32))[idx]), dtype=torch.float32),
                "hardneg_mode_apply_failure": torch.tensor(float(self.data.get("hardneg_mode_apply_failure", np.zeros((self.length,), dtype=np.float32))[idx]), dtype=torch.float32),
                "episode_index": torch.tensor(int(self.data["episode_index"][idx]), dtype=torch.long),
            }
            return out

        front = torch.from_numpy(self.data["front_rgb"][idx].astype(np.float32).transpose(2, 0, 1)) / 255.0
        wrist = torch.from_numpy(self.data["wrist_rgb"][idx].astype(np.float32).transpose(2, 0, 1)) / 255.0
        depth = torch.from_numpy(self.data["wrist_depth"][idx].astype(np.float32))
        if depth.ndim == 2:
            depth = depth.unsqueeze(0)
        out = {
            "front_rgb": front,
            "wrist_rgb": wrist,
            "wrist_depth": depth,
            "proprio": torch.from_numpy(self.data["proprio"][idx].astype(np.float32)),
            "gripper_context": torch.from_numpy(self.data["gripper_context"][idx].astype(np.float32)),
            "proxy_current_delta_basin_target": torch.from_numpy(self.data["proxy_current_delta_basin_target"][idx].astype(np.float32)),
            "current_dx_sign": torch.tensor(int(self.data["current_dx_sign"][idx]), dtype=torch.long),
            "current_dy_sign": torch.tensor(int(self.data["current_dy_sign"][idx]), dtype=torch.long),
            "current_dyaw_sign": torch.tensor(int(self.data["current_dyaw_sign"][idx]), dtype=torch.long),
            "basin_distance_bin": torch.tensor(int(self.data["basin_distance_bin"][idx]), dtype=torch.long),
            "substage_id": torch.tensor(int(self.data["substage_id"][idx]), dtype=torch.long),
            "contact_state": torch.tensor(int(self.data["contact_state"][idx]), dtype=torch.long),
            "stage_target_mode": torch.tensor(int(self.data["stage_target_mode"][idx]), dtype=torch.long),
            "candidate_actions_local": torch.from_numpy(self.data["candidate_actions_local"][idx].astype(np.float32)),
            "candidate_mask": torch.from_numpy(self.data["candidate_mask"][idx].astype(np.float32)),
            "candidate_oracle_score": torch.from_numpy(self.data["candidate_oracle_score"][idx].astype(np.float32)),
            "candidate_best_index": torch.tensor(int(self.data["candidate_best_index"][idx]), dtype=torch.long),
            "candidate_baseline_index": torch.tensor(int(self.data["candidate_baseline_index"][idx]), dtype=torch.long),
            "candidate_bad_index": torch.tensor(int(self.data.get("candidate_bad_index", np.full((self.length,), -1, dtype=np.int64))[idx]), dtype=torch.long),
            "sample_weight": torch.tensor(float(self.data.get("sample_weight", np.ones((self.length,), dtype=np.float32))[idx]), dtype=torch.float32),
            "is_shadow_hard_negative": torch.tensor(float(self.data.get("is_shadow_hard_negative", np.zeros((self.length,), dtype=np.float32))[idx]), dtype=torch.float32),
            "pred_worse_than_baseline": torch.tensor(float(self.data.get("pred_worse_than_baseline", np.zeros((self.length,), dtype=np.float32))[idx]), dtype=torch.float32),
            "pred_better_than_baseline": torch.tensor(float(self.data.get("pred_better_than_baseline", np.zeros((self.length,), dtype=np.float32))[idx]), dtype=torch.float32),
            "pred_has_yaw": torch.tensor(float(self.data.get("pred_has_yaw", np.zeros((self.length,), dtype=np.float32))[idx]), dtype=torch.float32),
            "oracle_has_yaw": torch.tensor(float(self.data.get("oracle_has_yaw", np.zeros((self.length,), dtype=np.float32))[idx]), dtype=torch.float32),
            "baseline_has_yaw": torch.tensor(float(self.data.get("baseline_has_yaw", np.zeros((self.length,), dtype=np.float32))[idx]), dtype=torch.float32),
            "pred_large_yaw_negative": torch.tensor(float(self.data.get("pred_large_yaw_negative", np.zeros((self.length,), dtype=np.float32))[idx]), dtype=torch.float32),
            "pred_large_yaw_positive": torch.tensor(float(self.data.get("pred_large_yaw_positive", np.zeros((self.length,), dtype=np.float32))[idx]), dtype=torch.float32),
            "pred_small_yaw_positive": torch.tensor(float(self.data.get("pred_small_yaw_positive", np.zeros((self.length,), dtype=np.float32))[idx]), dtype=torch.float32),
            "pred_yaw_bucket": torch.tensor(int(self.data.get("pred_yaw_bucket", np.zeros((self.length,), dtype=np.int64))[idx]), dtype=torch.long),
            "baseline_yaw_bucket": torch.tensor(int(self.data.get("baseline_yaw_bucket", np.zeros((self.length,), dtype=np.int64))[idx]), dtype=torch.long),
            "oracle_yaw_bucket": torch.tensor(int(self.data.get("oracle_yaw_bucket", np.zeros((self.length,), dtype=np.int64))[idx]), dtype=torch.long),
            "best_yaw_bucket": torch.tensor(int(self.data.get("best_yaw_bucket", np.zeros((self.length,), dtype=np.int64))[idx]), dtype=torch.long),
            "hard_episode_negative": torch.tensor(float(self.data.get("hard_episode_negative", np.zeros((self.length,), dtype=np.float32))[idx]), dtype=torch.float32),
            "mode_target_override": torch.tensor(int(self.data.get("mode_target_override", np.full((self.length,), -1, dtype=np.int64))[idx]), dtype=torch.long),
            "shadow_mode": torch.tensor(int(self.data.get("shadow_mode", np.full((self.length,), -1, dtype=np.int64))[idx]), dtype=torch.long),
            "hardneg_mode_keep_failure": torch.tensor(float(self.data.get("hardneg_mode_keep_failure", np.zeros((self.length,), dtype=np.float32))[idx]), dtype=torch.float32),
            "hardneg_mode_apply_failure": torch.tensor(float(self.data.get("hardneg_mode_apply_failure", np.zeros((self.length,), dtype=np.float32))[idx]), dtype=torch.float32),
            "episode_index": torch.tensor(int(self.data["episode_index"][idx]), dtype=torch.long),
        }
        return out


def load_handoff_model(ckpt_path: str, device: torch.device) -> StudentHandoffStateHeadV2:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model = StudentHandoffStateHeadV2().to(device)
    model.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=False)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


@torch.no_grad()
def extract_latent(handoff_model: StudentHandoffStateHeadV2, batch: dict[str, torch.Tensor], device: torch.device) -> torch.Tensor:
    out = handoff_model(
        front_rgb=batch["front_rgb"].to(device=device, dtype=torch.float32),
        wrist_rgb=batch["wrist_rgb"].to(device=device, dtype=torch.float32),
        wrist_depth=batch["wrist_depth"].to(device=device, dtype=torch.float32),
        proprio=batch["proprio"].to(device=device, dtype=torch.float32),
        gripper_context=batch["gripper_context"].to(device=device, dtype=torch.float32),
        proxy_current_delta_basin_target=batch["proxy_current_delta_basin_target"].to(device=device, dtype=torch.float32),
        current_dx_sign=batch["current_dx_sign"].to(device=device),
        current_dy_sign=batch["current_dy_sign"].to(device=device),
        current_dyaw_sign=batch["current_dyaw_sign"].to(device=device),
        basin_distance_bin=batch["basin_distance_bin"].to(device=device),
        substage_id=batch["substage_id"].to(device=device),
        contact_state=batch["contact_state"].to(device=device),
        stage_target_mode=batch["stage_target_mode"].to(device=device),
    )
    return out["latent"].detach()


@torch.no_grad()
def precompute_dataset_latents(
    dataset: CandidateRankingDataset,
    handoff_model: StudentHandoffStateHeadV2,
    device: torch.device,
    batch_size: int,
) -> None:
    if "handoff_latent" in dataset.data:
        return
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False)
    latents = []
    for batch in loader:
        latents.append(extract_latent(handoff_model, batch, device).cpu().numpy().astype(np.float32))
    dataset.data["handoff_latent"] = np.concatenate(latents, axis=0)


def mode_target_from_best_candidate(
    batch: dict[str, torch.Tensor],
    keep_yaw_abs: float,
    mode_apply_margin: float,
    device: torch.device,
) -> torch.Tensor:
    actions = batch["candidate_actions_local"].to(device=device, dtype=torch.float32)
    best_idx = batch["candidate_best_index"].to(device=device)
    baseline_idx = batch["candidate_baseline_index"].to(device=device)
    candidate_scores = batch["candidate_oracle_score"].to(device=device, dtype=torch.float32)
    row = torch.arange(actions.shape[0], device=device)
    best_yaw = torch.abs(actions[row, best_idx, 5])
    best_score = candidate_scores[row, best_idx]
    baseline_score = candidate_scores[row, baseline_idx]
    score_gap = best_score - baseline_score
    apply = (best_yaw > float(keep_yaw_abs)) & (score_gap > float(mode_apply_margin))
    return apply.long()


def mode_target_from_batch(
    batch: dict[str, torch.Tensor],
    keep_yaw_abs: float,
    mode_apply_margin: float,
    device: torch.device,
) -> torch.Tensor:
    default_target = mode_target_from_best_candidate(batch, keep_yaw_abs, mode_apply_margin, device)
    override = batch.get("mode_target_override")
    if override is None:
        return default_target
    override_t = override.to(device=device, dtype=torch.long)
    valid = override_t >= 0
    if not torch.any(valid):
        return default_target
    return torch.where(valid, override_t, default_target)


def mode_scope(mask: torch.Tensor, actions: torch.Tensor, mode_target: torch.Tensor, keep_yaw_abs: float) -> torch.Tensor:
    keep_scope = mask & (torch.abs(actions[:, :, 5]) <= float(keep_yaw_abs))
    apply_scope = mask & (torch.abs(actions[:, :, 5]) > float(keep_yaw_abs))
    selected = torch.where(mode_target.unsqueeze(1) > 0, apply_scope, keep_scope)
    empty = ~torch.any(selected, dim=1)
    selected[empty] = mask[empty]
    return selected


def _parse_csv_ints(text: str | None) -> set[int]:
    out: set[int] = set()
    if not text:
        return out
    for item in str(text).split(","):
        item = item.strip()
        if not item:
            continue
        out.add(int(item))
    return out


def _yaw_bucket_mask(actions: torch.Tensor, mask: torch.Tensor, keep_yaw_abs: float, small_yaw_abs: float, large_yaw_abs: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    yaw = torch.abs(actions[:, :, 5])
    no_yaw = mask & (yaw < float(keep_yaw_abs))
    small_yaw = mask & (yaw >= float(small_yaw_abs)) & (yaw < float(large_yaw_abs))
    large_yaw = mask & (yaw >= float(large_yaw_abs))
    return no_yaw, small_yaw, large_yaw


def _best_bucket_indices(oracle_scores: torch.Tensor, bucket_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    masked = oracle_scores.masked_fill(~bucket_mask, -1e9)
    best_scores, best_idx = masked.max(dim=1)
    valid = torch.any(bucket_mask, dim=1)
    best_idx = torch.where(valid, best_idx, torch.full_like(best_idx, -1))
    best_scores = torch.where(valid, best_scores, torch.full_like(best_scores, float("nan")))
    return best_idx, best_scores


def pairwise_ranking_loss(scores: torch.Tensor, oracle_scores: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    score_margin_limit = 20.0
    score_diff = scores.unsqueeze(2) - scores.unsqueeze(1)
    oracle_diff = oracle_scores.unsqueeze(2) - oracle_scores.unsqueeze(1)
    pair_mask = (oracle_diff > 1e-6) & mask.unsqueeze(2) & mask.unsqueeze(1)
    if not torch.any(pair_mask):
        return scores.sum() * 0.0
    gap = torch.clamp(oracle_diff, min=0.0)
    loss = F.softplus(-torch.clamp(score_diff, min=-score_margin_limit, max=score_margin_limit)) * gap
    denom = torch.clamp(gap[pair_mask].sum(), min=1e-6)
    return loss[pair_mask].sum() / denom


def compute_loss(
    model: StudentCandidateEvaluatorV2,
    handoff_model: StudentHandoffStateHeadV2,
    batch: dict[str, torch.Tensor],
    device: torch.device,
    keep_yaw_abs: float,
    mode_apply_margin: float,
    small_yaw_abs: float,
    large_yaw_abs: float,
    value_weight: float,
    pairwise_weight: float,
    best_ce_weight: float,
    mode_weight: float,
    baseline_pair_weight: float,
    hard_negative_weight: float,
    bad_yaw_pair_weight: float,
    large_yaw_negative_weight: float,
    yaw_positive_weight: float,
    large_yaw_positive_weight: float,
    small_vs_large_yaw_weight: float,
    hard_episode_negative_weight: float,
    hard_episode_indices: set[int],
) -> tuple[torch.Tensor, dict[str, float]]:
    score_margin_limit = 20.0
    if "handoff_latent" in batch:
        latent = batch["handoff_latent"].to(device=device, dtype=torch.float32)
    else:
        latent = extract_latent(handoff_model, batch, device)
    actions = batch["candidate_actions_local"].to(device=device, dtype=torch.float32)
    mask = batch["candidate_mask"].to(device=device, dtype=torch.float32) > 0.5
    out = model.forward_with_mode(
        handoff_latent=latent,
        proxy_current_delta_basin_target=batch["proxy_current_delta_basin_target"].to(device=device, dtype=torch.float32),
        candidate_actions_local=actions,
        candidate_mask=batch["candidate_mask"].to(device=device, dtype=torch.float32),
        yaw_aware_candidate_scope=batch["candidate_mask"].to(device=device, dtype=torch.float32),
    )
    oracle_scores = batch["candidate_oracle_score"].to(device=device, dtype=torch.float32)
    oracle_scores = torch.where(
        torch.isfinite(oracle_scores),
        oracle_scores,
        torch.full_like(oracle_scores, -1e9),
    )
    best_idx = batch["candidate_best_index"].to(device=device)
    baseline_idx = batch["candidate_baseline_index"].to(device=device)
    sample_weight = batch["sample_weight"].to(device=device, dtype=torch.float32)
    row_weight = torch.clamp(sample_weight, min=1e-3)
    episode_index = batch["episode_index"].to(device=device)

    row = torch.arange(actions.shape[0], device=device)
    mode_target = mode_target_from_batch(batch, keep_yaw_abs, mode_apply_margin, device)
    mode_logits = out["yaw_mode_logits"]
    generic_scores = out["candidate_scores"]
    keep_scores = out["candidate_scores_keep"]
    apply_scores = out["candidate_scores_apply"]
    generic_scores = torch.where(torch.isfinite(generic_scores), generic_scores, torch.full_like(generic_scores, -1e9))
    keep_scores = torch.where(torch.isfinite(keep_scores), keep_scores, torch.full_like(keep_scores, -1e9))
    apply_scores = torch.where(torch.isfinite(apply_scores), apply_scores, torch.full_like(apply_scores, -1e9))

    chosen_scores = torch.where(mode_target.unsqueeze(1) > 0, apply_scores, keep_scores)
    chosen_scope = mode_scope(mask, actions, mode_target, keep_yaw_abs)
    chosen_scope[row, best_idx] = True
    masked_scores = chosen_scores.masked_fill(~chosen_scope, -1e9)

    ce = F.cross_entropy(masked_scores, best_idx, reduction="none")
    ce_loss = (ce * sample_weight).sum() / torch.clamp(sample_weight.sum(), min=1e-6)

    oracle_mean = (oracle_scores * mask.float()).sum(dim=1) / torch.clamp(mask.float().sum(dim=1), min=1.0)
    oracle_std = torch.sqrt(
        torch.clamp(
            (((oracle_scores - oracle_mean.unsqueeze(1)) * mask.float()) ** 2).sum(dim=1) / torch.clamp(mask.float().sum(dim=1), min=1.0),
            min=1e-6,
        )
    )
    oracle_norm = ((oracle_scores - oracle_mean.unsqueeze(1)) / oracle_std.unsqueeze(1)).masked_fill(~mask, 0.0)
    value_loss = F.smooth_l1_loss(generic_scores[mask], oracle_norm[mask]) if torch.any(mask) else generic_scores.sum() * 0.0
    pair_loss = pairwise_ranking_loss(chosen_scores, oracle_scores, chosen_scope)
    mode_loss = F.cross_entropy(mode_logits, mode_target, reduction="none")
    mode_loss = (mode_loss * sample_weight).sum() / torch.clamp(sample_weight.sum(), min=1e-6)

    best_scores_generic = generic_scores[row, best_idx]
    baseline_scores_generic = generic_scores[row, baseline_idx]
    oracle_gap = oracle_scores[row, best_idx] - oracle_scores[row, baseline_idx]
    better_than_baseline_mask = oracle_gap > 1e-6
    if torch.any(better_than_baseline_mask):
        baseline_pair = F.softplus(-torch.clamp(best_scores_generic - baseline_scores_generic, min=-score_margin_limit, max=score_margin_limit))
        baseline_pair = baseline_pair * torch.clamp(oracle_gap, min=0.0)
        baseline_weights = row_weight[better_than_baseline_mask]
        baseline_pair_loss = (baseline_pair[better_than_baseline_mask] * baseline_weights).sum() / torch.clamp(baseline_weights.sum(), min=1e-6)
    else:
        baseline_pair_loss = generic_scores.sum() * 0.0

    bad_idx = batch["candidate_bad_index"].to(device=device)
    is_hard_neg = batch["is_shadow_hard_negative"].to(device=device, dtype=torch.float32) > 0.5
    hard_episode_mask = torch.zeros_like(is_hard_neg)
    if hard_episode_indices:
        hard_episode_mask = torch.zeros_like(is_hard_neg)
        for ep in hard_episode_indices:
            hard_episode_mask |= episode_index == int(ep)
    is_hard_episode = is_hard_neg & hard_episode_mask
    valid_bad = is_hard_neg & (bad_idx >= 0) & (bad_idx < generic_scores.shape[1])
    if torch.any(valid_bad):
        bad_scores = generic_scores[row, torch.clamp(bad_idx, 0, generic_scores.shape[1] - 1)]
        base_vs_bad = F.softplus(-torch.clamp(baseline_scores_generic - bad_scores, min=-score_margin_limit, max=score_margin_limit))
        best_vs_bad = F.softplus(-torch.clamp(best_scores_generic - bad_scores, min=-score_margin_limit, max=score_margin_limit))
        yaw_bad_mask = valid_bad & (batch["pred_has_yaw"].to(device=device, dtype=torch.float32) > 0.5) & (batch["oracle_has_yaw"].to(device=device, dtype=torch.float32) < 0.5)
        hard_weights = torch.ones_like(base_vs_bad)
        hard_weights = torch.where(yaw_bad_mask, hard_weights * float(max(bad_yaw_pair_weight, 1.0)), hard_weights)
        hard_weights = torch.where(is_hard_episode & valid_bad, hard_weights * float(max(hard_episode_negative_weight, 1.0)), hard_weights)
        hard_negative_weights = row_weight[valid_bad] * hard_weights[valid_bad]
        hard_negative_loss = (((base_vs_bad + best_vs_bad) * 0.5)[valid_bad] * hard_negative_weights).sum() / torch.clamp(hard_negative_weights.sum(), min=1e-6)
    else:
        hard_negative_loss = generic_scores.sum() * 0.0

    large_yaw_negative = batch["pred_large_yaw_negative"].to(device=device, dtype=torch.float32) > 0.5
    if torch.any(large_yaw_negative & valid_bad):
        bad_scores = generic_scores[row, torch.clamp(bad_idx, 0, generic_scores.shape[1] - 1)]
        large_yaw_loss_per_row = (
            F.softplus(-torch.clamp(baseline_scores_generic - bad_scores, min=-score_margin_limit, max=score_margin_limit))
            + F.softplus(-torch.clamp(best_scores_generic - bad_scores, min=-score_margin_limit, max=score_margin_limit))
        )
        weights_large = row_weight[large_yaw_negative & valid_bad]
        large_yaw_loss = (large_yaw_loss_per_row[large_yaw_negative & valid_bad] * weights_large).sum() / torch.clamp(weights_large.sum(), min=1e-6)
    else:
        large_yaw_loss = generic_scores.sum() * 0.0

    large_yaw_positive = batch["pred_large_yaw_positive"].to(device=device, dtype=torch.float32) > 0.5
    if torch.any(large_yaw_positive):
        actions_yaw = torch.abs(actions[:, :, 5])
        small_bucket = (actions_yaw >= float(small_yaw_abs)) & (actions_yaw < float(large_yaw_abs)) & mask
        small_idx, small_scores = _best_bucket_indices(oracle_scores, small_bucket)
        valid_pos = large_yaw_positive & (small_idx >= 0)
        if torch.any(valid_pos):
            pred_large_scores = generic_scores[row, torch.clamp(bad_idx, 0, generic_scores.shape[1] - 1)]
            small_best_scores = small_scores
            large_positive_loss = F.softplus(-torch.clamp(pred_large_scores - small_best_scores, min=-score_margin_limit, max=score_margin_limit))
            weights_pos = row_weight[valid_pos]
            large_positive_loss = (large_positive_loss[valid_pos] * weights_pos).sum() / torch.clamp(weights_pos.sum(), min=1e-6)
        else:
            large_positive_loss = generic_scores.sum() * 0.0
    else:
        large_positive_loss = generic_scores.sum() * 0.0

    no_yaw_bucket, small_yaw_bucket, large_yaw_bucket = _yaw_bucket_mask(actions, mask, keep_yaw_abs, small_yaw_abs, large_yaw_abs)
    small_best_idx, small_best_scores = _best_bucket_indices(oracle_scores, small_yaw_bucket)
    large_best_idx, large_best_scores = _best_bucket_indices(oracle_scores, large_yaw_bucket)
    small_large_valid = (small_best_idx >= 0) & (large_best_idx >= 0) & ((small_best_scores - large_best_scores) > 1e-6)
    if torch.any(small_large_valid):
        small_scores = generic_scores[row, torch.clamp(small_best_idx, 0, generic_scores.shape[1] - 1)]
        large_scores = generic_scores[row, torch.clamp(large_best_idx, 0, generic_scores.shape[1] - 1)]
        small_vs_large_loss = F.softplus(-torch.clamp(small_scores - large_scores, min=-score_margin_limit, max=score_margin_limit))
        weights_svl = row_weight[small_large_valid]
        small_vs_large_loss = (small_vs_large_loss[small_large_valid] * weights_svl).sum() / torch.clamp(weights_svl.sum(), min=1e-6)
    else:
        small_vs_large_loss = generic_scores.sum() * 0.0

    yaw_positive = batch["pred_small_yaw_positive"].to(device=device, dtype=torch.float32) > 0.5
    if torch.any(yaw_positive):
        yaw_pos_per_row = F.softplus(-torch.clamp(best_scores_generic - baseline_scores_generic, min=-score_margin_limit, max=score_margin_limit))
        weights_yaw = row_weight[yaw_positive]
        yaw_positive_loss = (yaw_pos_per_row[yaw_positive] * weights_yaw).sum() / torch.clamp(weights_yaw.sum(), min=1e-6)
    else:
        yaw_positive_loss = generic_scores.sum() * 0.0

    loss = (
        float(best_ce_weight) * ce_loss
        + float(pairwise_weight) * pair_loss
        + float(value_weight) * value_loss
        + float(mode_weight) * mode_loss
        + float(baseline_pair_weight) * baseline_pair_loss
        + float(hard_negative_weight) * hard_negative_loss
        + float(large_yaw_negative_weight) * large_yaw_loss
        + float(yaw_positive_weight) * yaw_positive_loss
        + float(large_yaw_positive_weight) * large_positive_loss
        + float(small_vs_large_yaw_weight) * small_vs_large_loss
    )

    with torch.no_grad():
        pred_scores = chosen_scores.masked_fill(~chosen_scope, -1e9)
        pred_idx = torch.argmax(pred_scores, dim=1)
        topk = torch.topk(pred_scores, k=min(3, pred_scores.shape[1]), dim=1).indices
        pred_score = oracle_scores[row, pred_idx]
        base_score = oracle_scores[row, baseline_idx]
        best_score = oracle_scores[row, best_idx]
        pred_better = pred_score > (base_score + 1e-6)
        pred_worse = pred_score < (base_score - 1e-6)
        yaw_candidates = torch.any((torch.abs(actions[:, :, 5]) > float(keep_yaw_abs)) & mask, dim=1)
        pred_has_yaw = torch.abs(actions[row, pred_idx, 5]) > float(keep_yaw_abs)
        oracle_has_yaw = torch.abs(actions[row, best_idx, 5]) > float(keep_yaw_abs)
        metrics = {
            "loss": float(loss.item()),
            "top1_recall": float(torch.mean((pred_idx == best_idx).float()).item()),
            "top3_recall": float(torch.mean(torch.any(topk == best_idx.unsqueeze(1), dim=1).float()).item()),
            "selected_improves_rate": float(torch.mean(pred_better.float()).item()),
            "negative_selection_rate": float(torch.mean(pred_worse.float()).item()),
            "regret_delta_mean": float(torch.mean(pred_score - base_score).item()),
            "oracle_gap_mean": float(torch.mean(best_score - base_score).item()),
            "mode_acc": float(torch.mean((torch.argmax(mode_logits, dim=1) == mode_target).float()).item()),
            "yaw_candidate_selection_rate": float(torch.mean(pred_has_yaw[yaw_candidates].float()).item()) if torch.any(yaw_candidates) else 0.0,
            "oracle_yaw_candidate_rate": float(torch.mean(oracle_has_yaw[yaw_candidates].float()).item()) if torch.any(yaw_candidates) else 0.0,
            "rows_with_yaw_candidates": int(torch.sum(yaw_candidates).item()),
            "baseline_pair_loss": float(baseline_pair_loss.item()),
            "hard_negative_loss": float(hard_negative_loss.item()),
            "large_yaw_negative_loss": float(large_yaw_loss.item()),
            "yaw_positive_loss": float(yaw_positive_loss.item()),
            "large_yaw_positive_loss": float(large_positive_loss.item()),
            "small_vs_large_yaw_loss": float(small_vs_large_loss.item()),
            "hard_negative_rows": int(torch.sum(valid_bad).item()),
            "hard_episode_rows": int(torch.sum(is_hard_episode & valid_bad).item()),
            "large_yaw_negative_rows": int(torch.sum(large_yaw_negative & valid_bad).item()),
            "large_yaw_positive_rows": int(torch.sum(large_yaw_positive).item()),
            "yaw_positive_rows": int(torch.sum(yaw_positive).item()),
        }
    return loss, metrics


@torch.no_grad()
def evaluate(
    model: StudentCandidateEvaluatorV2,
    handoff_model: StudentHandoffStateHeadV2,
    loader: DataLoader,
    device: torch.device,
    keep_yaw_abs: float,
    mode_apply_margin: float,
    small_yaw_abs: float,
    large_yaw_abs: float,
    value_weight: float,
    pairwise_weight: float,
    best_ce_weight: float,
    mode_weight: float,
    baseline_pair_weight: float,
    hard_negative_weight: float,
    bad_yaw_pair_weight: float,
    large_yaw_negative_weight: float,
    yaw_positive_weight: float,
    large_yaw_positive_weight: float,
    small_vs_large_yaw_weight: float,
    hard_episode_negative_weight: float,
    hard_episode_indices: set[int],
) -> dict[str, float]:
    model.eval()
    agg: dict[str, float] = {}
    count = 0
    for batch in loader:
        _, metrics = compute_loss(
            model,
            handoff_model,
            batch,
            device,
            keep_yaw_abs=keep_yaw_abs,
            mode_apply_margin=mode_apply_margin,
            small_yaw_abs=small_yaw_abs,
            large_yaw_abs=large_yaw_abs,
            value_weight=value_weight,
            pairwise_weight=pairwise_weight,
            best_ce_weight=best_ce_weight,
            mode_weight=mode_weight,
            baseline_pair_weight=baseline_pair_weight,
            hard_negative_weight=hard_negative_weight,
            bad_yaw_pair_weight=bad_yaw_pair_weight,
            large_yaw_negative_weight=large_yaw_negative_weight,
            yaw_positive_weight=yaw_positive_weight,
            large_yaw_positive_weight=large_yaw_positive_weight,
            small_vs_large_yaw_weight=small_vs_large_yaw_weight,
            hard_episode_negative_weight=hard_episode_negative_weight,
            hard_episode_indices=hard_episode_indices,
        )
        bsz = batch["candidate_best_index"].shape[0]
        count += bsz
        for k, v in metrics.items():
            agg[k] = agg.get(k, 0.0) + float(v) * bsz
    if count <= 0:
        return {k: 0.0 for k in ("loss", "top1_recall", "top3_recall", "selected_improves_rate", "negative_selection_rate", "regret_delta_mean", "oracle_gap_mean", "mode_acc", "yaw_candidate_selection_rate", "oracle_yaw_candidate_rate", "rows_with_yaw_candidates")}
    return {k: float(v / count) for k, v in agg.items()}


def subset_loader(dataset: CandidateRankingDataset, indices: Iterable[int], batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(Subset(dataset, list(indices)), batch_size=batch_size, shuffle=shuffle, num_workers=0, drop_last=False)


def loeo_splits(dataset: CandidateRankingDataset) -> list[tuple[int, list[int], list[int]]]:
    eps = np.asarray(dataset.data["episode_index"], dtype=np.int64)
    uniq = sorted(int(x) for x in np.unique(eps))
    splits = []
    for heldout in uniq:
        train_idx = np.where(eps != heldout)[0].tolist()
        val_idx = np.where(eps == heldout)[0].tolist()
        if train_idx and val_idx:
            splits.append((heldout, train_idx, val_idx))
    return splits


def train_one(
    dataset: CandidateRankingDataset,
    train_idx: list[int],
    val_idx: list[int],
    args: argparse.Namespace,
    device: torch.device,
    handoff_model: StudentHandoffStateHeadV2,
) -> tuple[StudentCandidateEvaluatorV2, dict[str, float], dict[str, float]]:
    model = StudentCandidateEvaluatorV2(yaw_mode_classes=2).to(device)
    if args.mode_input_path:
        model.set_mode_input_path(args.mode_input_path)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    train_loader = subset_loader(dataset, train_idx, args.batch_size, True)
    val_loader = subset_loader(dataset, val_idx, args.batch_size, False)
    best_val = None
    best_state = None
    best_metrics: dict[str, float] = {}
    for epoch in range(args.epochs):
        model.train()
        for batch in train_loader:
            opt.zero_grad(set_to_none=True)
            loss, _ = compute_loss(
                model, handoff_model, batch, device,
                keep_yaw_abs=args.keep_yaw_abs,
                mode_apply_margin=args.mode_apply_margin,
                small_yaw_abs=args.small_yaw_abs,
                large_yaw_abs=args.large_yaw_abs,
                value_weight=args.value_weight,
                pairwise_weight=args.pairwise_weight,
                best_ce_weight=args.best_ce_weight,
                mode_weight=args.mode_weight,
                baseline_pair_weight=args.baseline_pair_weight,
                hard_negative_weight=args.hard_negative_weight,
                bad_yaw_pair_weight=args.bad_yaw_pair_weight,
                large_yaw_negative_weight=args.large_yaw_negative_weight,
                yaw_positive_weight=args.yaw_positive_weight,
                large_yaw_positive_weight=args.large_yaw_positive_weight,
                small_vs_large_yaw_weight=args.small_vs_large_yaw_weight,
                hard_episode_negative_weight=args.hard_episode_negative_weight,
                hard_episode_indices=args.hard_episode_indices,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
        val_metrics = evaluate(
            model, handoff_model, val_loader, device,
            keep_yaw_abs=args.keep_yaw_abs,
            mode_apply_margin=args.mode_apply_margin,
            small_yaw_abs=args.small_yaw_abs,
            large_yaw_abs=args.large_yaw_abs,
            value_weight=args.value_weight,
            pairwise_weight=args.pairwise_weight,
            best_ce_weight=args.best_ce_weight,
            mode_weight=args.mode_weight,
            baseline_pair_weight=args.baseline_pair_weight,
            hard_negative_weight=args.hard_negative_weight,
            bad_yaw_pair_weight=args.bad_yaw_pair_weight,
            large_yaw_negative_weight=args.large_yaw_negative_weight,
            yaw_positive_weight=args.yaw_positive_weight,
            large_yaw_positive_weight=args.large_yaw_positive_weight,
            small_vs_large_yaw_weight=args.small_vs_large_yaw_weight,
            hard_episode_negative_weight=args.hard_episode_negative_weight,
            hard_episode_indices=args.hard_episode_indices,
        )
        score = val_metrics["selected_improves_rate"] + 0.25 * val_metrics["top1_recall"] + 0.10 * val_metrics["top3_recall"] + 0.10 * max(val_metrics["regret_delta_mean"], 0.0)
        if best_val is None or score > best_val:
            best_val = score
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            best_metrics = dict(val_metrics)
    assert best_state is not None
    model.load_state_dict(best_state, strict=True)
    train_metrics = evaluate(
        model, handoff_model, train_loader, device,
        keep_yaw_abs=args.keep_yaw_abs,
        mode_apply_margin=args.mode_apply_margin,
        small_yaw_abs=args.small_yaw_abs,
        large_yaw_abs=args.large_yaw_abs,
        value_weight=args.value_weight,
        pairwise_weight=args.pairwise_weight,
        best_ce_weight=args.best_ce_weight,
        mode_weight=args.mode_weight,
        baseline_pair_weight=args.baseline_pair_weight,
        hard_negative_weight=args.hard_negative_weight,
        bad_yaw_pair_weight=args.bad_yaw_pair_weight,
        large_yaw_negative_weight=args.large_yaw_negative_weight,
        yaw_positive_weight=args.yaw_positive_weight,
        large_yaw_positive_weight=args.large_yaw_positive_weight,
        small_vs_large_yaw_weight=args.small_vs_large_yaw_weight,
        hard_episode_negative_weight=args.hard_episode_negative_weight,
        hard_episode_indices=args.hard_episode_indices,
    )
    return model, train_metrics, best_metrics


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_npz", action="append", required=True)
    ap.add_argument("--handoff_ckpt", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--keep_yaw_abs", type=float, default=0.02)
    ap.add_argument("--mode_apply_margin", type=float, default=0.05)
    ap.add_argument("--small_yaw_abs", type=float, default=0.05)
    ap.add_argument("--large_yaw_abs", type=float, default=0.09)
    ap.add_argument("--pairwise_weight", type=float, default=0.6)
    ap.add_argument("--value_weight", type=float, default=0.2)
    ap.add_argument("--best_ce_weight", type=float, default=0.6)
    ap.add_argument("--mode_weight", type=float, default=0.2)
    ap.add_argument("--baseline_pair_weight", type=float, default=0.5)
    ap.add_argument("--hard_negative_weight", type=float, default=1.0)
    ap.add_argument("--bad_yaw_pair_weight", type=float, default=2.0)
    ap.add_argument("--large_yaw_negative_weight", type=float, default=2.0)
    ap.add_argument("--large_yaw_positive_weight", type=float, default=1.0)
    ap.add_argument("--small_vs_large_yaw_weight", type=float, default=1.5)
    ap.add_argument("--hard_episode_negative_weight", type=float, default=3.0)
    ap.add_argument("--hard_episode_indices", type=str, default="17")
    ap.add_argument("--yaw_positive_weight", type=float, default=0.5)
    ap.add_argument("--mode_input_path", type=str, default="summary_only", choices=["summary_only", "hybrid"])
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()
    args.hard_episode_indices = _parse_csv_ints(args.hard_episode_indices)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = CandidateRankingDataset(args.dataset_npz)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    handoff_model = load_handoff_model(args.handoff_ckpt, device)
    precompute_dataset_latents(dataset, handoff_model, device, args.batch_size)

    loeo_results = []
    for heldout_ep, train_idx, val_idx in loeo_splits(dataset):
        model, train_metrics, val_metrics = train_one(dataset, train_idx, val_idx, args, device, handoff_model)
        loeo_results.append({
            "heldout_episode": int(heldout_ep),
            "train_rows": int(len(train_idx)),
            "val_rows": int(len(val_idx)),
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
        })

    all_idx = list(range(len(dataset)))
    final_model, final_train_metrics, _ = train_one(dataset, all_idx, all_idx, args, device, handoff_model)
    ckpt_path = output_dir / "student_candidate_evaluator_v4b_best.pt"
    torch.save(
        {
            "model_state_dict": final_model.state_dict(),
            "handoff_state_ckpt": str(args.handoff_ckpt),
            "yaw_mode_num_classes": 2,
            "yaw_keep_abs": float(args.keep_yaw_abs),
            "mode_apply_margin": float(args.mode_apply_margin),
            "mode_input_path": str(args.mode_input_path),
            "shadow_policy": "generic_ranker_v4b",
            "train_args": vars(args),
            "final_train_metrics": final_train_metrics,
            "loeo_results": loeo_results,
        },
        ckpt_path,
    )

    def mean_metric(path: str) -> float:
        vals = [float(r["val_metrics"].get(path, 0.0)) for r in loeo_results]
        return float(np.mean(vals)) if vals else 0.0

    report = {
        "dataset_npz": list(args.dataset_npz),
        "rows": int(len(dataset)),
        "episodes": int(np.unique(dataset.data["episode_index"]).size),
        "checkpoint": str(ckpt_path),
        "loeo_mean_top1_recall": mean_metric("top1_recall"),
        "loeo_mean_top3_recall": mean_metric("top3_recall"),
        "loeo_mean_selected_improves_rate": mean_metric("selected_improves_rate"),
        "loeo_mean_negative_selection_rate": mean_metric("negative_selection_rate"),
        "loeo_mean_regret_delta": mean_metric("regret_delta_mean"),
        "loeo_mean_mode_acc": mean_metric("mode_acc"),
        "loeo_mean_yaw_candidate_selection_rate": mean_metric("yaw_candidate_selection_rate"),
        "loeo_mean_oracle_yaw_candidate_rate": mean_metric("oracle_yaw_candidate_rate"),
        "final_train_metrics": final_train_metrics,
        "loeo_results": loeo_results,
    }
    report_path = output_dir / "alignment_v4b_candidate_ranker_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
