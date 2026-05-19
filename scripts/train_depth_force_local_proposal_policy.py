#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from prismatic.models.depth_force_local_proposal_policy import (
    DepthForceLocalProposalPolicy,
    LocalProposalActionScale,
)
from prismatic.vla.datasets.depth_force_local_proposal_dataset import MODE_TO_ID, DepthForceLocalProposalDataset
from local_proposal_utils import evaluate_state_conditioned_proposals


ACTION_SCALE = torch.tensor([0.008, 0.008, 0.006, 0.06, 0.06, 0.12], dtype=torch.float32)


def _resolve_model_kwargs(input_mode: str) -> dict[str, bool]:
    mode = str(input_mode).strip().lower()
    if mode == "proprio_planner":
        return {
            "use_front_rgb": False,
            "use_wrist_rgb": False,
            "use_wrist_depth": False,
            "use_force": False,
            "use_proprio": True,
            "use_stage": False,
            "use_contact": False,
            "use_scalar": False,
            "use_candidate_depth_context": False,
            "use_candidate_force_context": False,
        }
    if mode == "force_proprio_planner":
        return {
            "use_front_rgb": False,
            "use_wrist_rgb": False,
            "use_wrist_depth": False,
            "use_force": True,
            "use_proprio": True,
            "use_stage": False,
            "use_contact": False,
            "use_scalar": False,
            "use_candidate_depth_context": False,
            "use_candidate_force_context": True,
        }
    if mode == "depth_proprio_planner":
        return {
            "use_front_rgb": False,
            "use_wrist_rgb": False,
            "use_wrist_depth": True,
            "use_force": False,
            "use_proprio": True,
            "use_stage": False,
            "use_contact": False,
            "use_scalar": False,
            "use_candidate_depth_context": True,
            "use_candidate_force_context": False,
        }
    if mode == "depth_force_proprio_planner":
        return {
            "use_front_rgb": False,
            "use_wrist_rgb": False,
            "use_wrist_depth": True,
            "use_force": True,
            "use_proprio": True,
            "use_stage": False,
            "use_contact": False,
            "use_scalar": False,
            "use_candidate_depth_context": True,
            "use_candidate_force_context": True,
        }
    raise ValueError(
        f"unknown input_mode={input_mode!r}; expected one of "
        "'proprio_planner', 'force_proprio_planner', 'depth_proprio_planner', 'depth_force_proprio_planner'"
    )


def _weighted_l1(a: torch.Tensor, b: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    scale = scale.to(device=a.device, dtype=a.dtype).view(*([1] * (a.ndim - 1)), -1)
    return torch.sum(torch.abs((a - b) / torch.clamp(scale, min=1e-6)), dim=-1)


def _pairwise_weighted_l1(a: torch.Tensor, b: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    scale = scale.to(device=a.device, dtype=a.dtype).view(1, 1, -1)
    return torch.sum(torch.abs((a[:, :, None, :] - b[:, None, :, :]) / torch.clamp(scale, min=1e-6)), dim=-1)


def _safe_target_mask(sample: dict[str, torch.Tensor]) -> torch.Tensor:
    mask = sample["proposal_pareto_mask"] > 0.5
    if torch.any(mask):
        return mask
    mask = sample["proposal_budget_mask"] > 0.5
    if torch.any(mask):
        return mask
    mask = torch.zeros_like(sample["proposal_pareto_mask"], dtype=torch.bool)
    mask[int(sample["proposal_best_safe_index"].item())] = True
    return mask


def _build_frontier_targets(sample: dict[str, torch.Tensor]) -> torch.Tensor:
    actions = sample["proposal_actions_local"]
    mask = _safe_target_mask(sample)
    targets = actions[mask]
    if targets.numel() == 0:
        targets = actions[sample["proposal_best_safe_index"].long().view(1)]
    return targets


def _quality_targets(sample: dict[str, torch.Tensor]) -> torch.Tensor:
    actions = sample["proposal_actions_local"]
    bank_mask = _safe_target_mask(sample)
    if not torch.any(bank_mask):
        bank_mask = torch.zeros_like(sample["proposal_pareto_mask"], dtype=torch.bool)
        bank_mask[int(sample["proposal_best_safe_index"].item())] = True
    return bank_mask.float()


def _mode_target(sample: dict[str, torch.Tensor]) -> torch.Tensor:
    return sample["proposal_target_mode_id"].long()


def _pairwise_rank_loss(scores: torch.Tensor, utilities: torch.Tensor, *, margin: float = 0.0) -> torch.Tensor:
    scores = scores.view(-1)
    utilities = utilities.view(-1)
    if scores.numel() <= 1:
        return scores.sum() * 0.0
    diff_u = utilities[:, None] - utilities[None, :]
    diff_s = scores[:, None] - scores[None, :]
    mask = torch.triu(diff_u > 1e-6, diagonal=1)
    if not torch.any(mask):
        return scores.sum() * 0.0
    loss = F.softplus(-(diff_s - float(margin)))
    return loss[mask].mean()


def _pareto_mask_from_gain_risk(gain: torch.Tensor, risk_delta: torch.Tensor) -> torch.Tensor:
    gain = gain.view(-1)
    risk_delta = risk_delta.view(-1)
    n = gain.numel()
    if n == 0:
        return torch.zeros((0,), dtype=torch.bool, device=gain.device)
    mask = torch.ones((n,), dtype=torch.bool, device=gain.device)
    eps = 1e-6
    for i in range(n):
        if not bool(mask[i].item()):
            continue
        dominated = (
            (gain >= gain[i] - eps)
            & (risk_delta <= risk_delta[i] + eps)
            & ((gain > gain[i] + eps) | (risk_delta < risk_delta[i] - eps))
        )
        dominated[i] = False
        if torch.any(dominated):
            mask[i] = False
    return mask


def _freeze_score_only(model: DepthForceLocalProposalPolicy, *, observability_tune: bool = False) -> None:
    for name, param in model.named_parameters():
        param.requires_grad = False
        if observability_tune:
            if (
                name.startswith("depth_encoder")
                or name.startswith("force_encoder")
                or name.startswith("proprio_encoder")
                or name.startswith("state_trunk")
                or name.startswith("proposal_candidate_encoder")
                or name.startswith("proposal_context_encoder")
                or name.startswith("proposal_score_head")
                or name.startswith("multi_head_context_encoder")
                or name.startswith("multi_head_score_head")
            ):
                param.requires_grad = True
        elif name.startswith("proposal_candidate_encoder") or name.startswith("proposal_score_head"):
            param.requires_grad = True


def _load_init_checkpoint(path: str | None) -> dict | None:
    if not path:
        return None
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if "model_state_dict" not in ckpt:
        raise KeyError(f"checkpoint {path} is missing model_state_dict")
    return ckpt


def _summarize_state_dict_load(
    model: DepthForceLocalProposalPolicy,
    loaded_state_dict: dict[str, torch.Tensor],
) -> dict[str, list[str]]:
    model_keys = set(model.state_dict().keys())
    loaded_keys = set(loaded_state_dict.keys())
    missing_keys = sorted(model_keys - loaded_keys)
    unexpected_keys = sorted(loaded_keys - model_keys)
    intersect_keys = sorted(model_keys & loaded_keys)
    newly_initialized_keys = list(missing_keys)
    return {
        "missing_keys": missing_keys,
        "unexpected_keys": unexpected_keys,
        "loaded_keys": intersect_keys,
        "newly_initialized_keys": newly_initialized_keys,
    }


def _validate_trainable_parameters(model: DepthForceLocalProposalPolicy) -> list[str]:
    trainable_names = [name for name, param in model.named_parameters() if param.requires_grad]
    forbidden_prefixes = (
        "proposal_head",
        "proposal_action_head",
        "proposal_mean_head",
        "proposal_scale_head",
    )
    violations = [name for name in trainable_names if name.startswith(forbidden_prefixes)]
    if violations:
        raise RuntimeError(
            "proposal generator parameters must remain frozen during score/observability tuning; "
            f"found trainable forbidden parameters: {violations}"
        )
    return trainable_names


def _multi_head_score_loss(
    sample: dict[str, torch.Tensor],
    outputs: dict[str, torch.Tensor],
    *,
    score_utility_alpha: float,
    score_safe_risk_budget: float,
    yaw_presence_threshold: float,
    yaw_match_tol: float,
    best_safe_ce_weight: float,
    pareto_bce_weight: float,
    yaw_bce_weight: float,
    risk_safe_bce_weight: float,
    geometry_reg_weight: float,
    utility_rank_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    multi = outputs.get("multi_head_scores")
    if multi is None:
        raise KeyError("multi-head score training requires model output 'multi_head_scores'")
    # Shape: (K, 5) for one row.
    if multi.ndim != 2 or multi.shape[-1] != 5:
        raise ValueError(f"expected row multi_head_scores shape (K,5), got {tuple(multi.shape)}")
    device = multi.device
    dtype = multi.dtype
    geom_gain = sample["proposal_geometry_gain"].to(device=device, dtype=dtype)
    risk_delta = sample["proposal_risk_delta"].to(device=device, dtype=dtype)
    pareto = (sample["proposal_pareto_mask"].to(device=device, dtype=dtype) > 0.5).float()
    best_safe_idx = sample["proposal_best_safe_index"].long().to(device=device).view(1)
    actions = sample["proposal_actions_local"].to(device=device, dtype=dtype)
    best_safe_yaw_abs = torch.abs(actions[best_safe_idx.item(), 5])
    target_yaw = sample["proposal_target_delta_local"].to(device=device, dtype=dtype)[5]
    yaw_match = (torch.abs(torch.abs(actions[:, 5]) - best_safe_yaw_abs) <= float(yaw_match_tol)).float()
    if torch.abs(target_yaw) <= float(yaw_presence_threshold):
        yaw_match = torch.zeros_like(yaw_match)
    risk_safe = (risk_delta <= float(score_safe_risk_budget)).float()
    utility = geom_gain - float(score_utility_alpha) * torch.clamp(risk_delta, min=0.0)

    best_safe_score = multi[:, 0]
    pareto_score = multi[:, 1]
    yaw_score = multi[:, 2]
    risk_score = multi[:, 3]
    geom_score = multi[:, 4]
    final_score = best_safe_score + pareto_score + yaw_score + geom_score - risk_score

    best_safe_ce = F.cross_entropy(best_safe_score.unsqueeze(0), best_safe_idx)
    pareto_bce = F.binary_cross_entropy_with_logits(pareto_score, pareto)
    yaw_bce = F.binary_cross_entropy_with_logits(yaw_score, yaw_match)
    risk_bce = F.binary_cross_entropy_with_logits(risk_score, risk_safe)
    geom_target = (geom_gain - torch.mean(geom_gain)) / torch.clamp(torch.std(geom_gain, unbiased=False), min=1e-4)
    geometry_reg = F.smooth_l1_loss(geom_score, geom_target.detach())
    utility_rank = _pairwise_rank_loss(final_score, utility, margin=0.0)

    loss = (
        float(best_safe_ce_weight) * best_safe_ce
        + float(pareto_bce_weight) * pareto_bce
        + float(yaw_bce_weight) * yaw_bce
        + float(risk_safe_bce_weight) * risk_bce
        + float(geometry_reg_weight) * geometry_reg
        + float(utility_rank_weight) * utility_rank
    )
    selected = int(torch.argmax(final_score).item())
    ranks = torch.argsort(torch.argsort(-final_score)) + 1
    utility_best = int(torch.argmax(utility).item())
    yaw_positive = torch.nonzero(yaw_match > 0.5).view(-1)
    yaw_rank = float("nan")
    if yaw_positive.numel() > 0:
        yaw_rank = float(torch.min(ranks[yaw_positive]).detach().cpu().item())
    stats = {
        "multi_head_loss": float(loss.detach().cpu().item()),
        "multi_best_safe_ce": float(best_safe_ce.detach().cpu().item()),
        "multi_pareto_bce": float(pareto_bce.detach().cpu().item()),
        "multi_yaw_bce": float(yaw_bce.detach().cpu().item()),
        "multi_risk_safe_bce": float(risk_bce.detach().cpu().item()),
        "multi_geometry_reg": float(geometry_reg.detach().cpu().item()),
        "multi_utility_rank": float(utility_rank.detach().cpu().item()),
        "multi_selected_best_safe_hit": float(selected == int(best_safe_idx.item())),
        "multi_selected_pareto_hit": float(bool(pareto[selected].detach().cpu().item() > 0.5)),
        "multi_selected_yaw_match": float(bool(yaw_match[selected].detach().cpu().item() > 0.5)),
        "multi_selected_risk_safe": float(bool(risk_safe[selected].detach().cpu().item() > 0.5)),
        "multi_selected_geom_gain": float(geom_gain[selected].detach().cpu().item()),
        "multi_selected_risk_delta": float(risk_delta[selected].detach().cpu().item()),
        "multi_best_safe_rank": float(ranks[int(best_safe_idx.item())].detach().cpu().item()),
        "multi_utility_best_rank": float(ranks[utility_best].detach().cpu().item()),
        "multi_yaw_match_rank": yaw_rank,
    }
    return loss, stats


def _multi_head_final_score(
    outputs: dict[str, torch.Tensor],
    *,
    w_safe: float,
    w_pareto: float,
    w_yaw: float,
    w_geom: float,
    w_risk: float,
) -> torch.Tensor:
    multi = outputs["multi_head_scores"]
    return (
        float(w_safe) * multi[..., 0]
        + float(w_pareto) * multi[..., 1]
        + float(w_yaw) * multi[..., 2]
        + float(w_geom) * multi[..., 4]
        - float(w_risk) * multi[..., 3]
    )


def _multi_head_layered_score(
    outputs: dict[str, torch.Tensor],
    *,
    mode: str,
    w_safe: float,
    w_pareto: float,
    w_yaw: float,
    w_geom: float,
    w_risk: float,
) -> torch.Tensor:
    mode = str(mode).lower()
    if mode == "weighted_multi":
        return _multi_head_final_score(
            outputs,
            w_safe=w_safe,
            w_pareto=w_pareto,
            w_yaw=w_yaw,
            w_geom=w_geom,
            w_risk=w_risk,
        )
    multi = outputs["multi_head_scores"]
    pareto_prob = torch.sigmoid(multi[..., 1])
    safe_prob = torch.sigmoid(multi[..., 0])
    yaw_prob = torch.sigmoid(multi[..., 2])
    risk_prob = torch.sigmoid(multi[..., 3])
    geom_prob = torch.sigmoid(multi[..., 4])
    if mode == "pareto_then_best_safe":
        base = safe_prob + 0.5 * geom_prob + 0.25 * yaw_prob - 0.5 * risk_prob
    elif mode == "pareto_then_geometry":
        base = geom_prob + 0.5 * safe_prob + 0.25 * yaw_prob - 0.5 * risk_prob
    else:
        base = safe_prob + pareto_prob + 0.5 * yaw_prob + 0.75 * geom_prob - 0.5 * risk_prob
    gate = pareto_prob >= 0.5
    any_gate = torch.any(gate, dim=-1, keepdim=True)
    fallback_k = min(3, base.shape[-1])
    topk_idx = torch.topk(pareto_prob, k=fallback_k, dim=-1).indices
    fallback_mask = torch.zeros_like(gate, dtype=torch.bool).scatter(-1, topk_idx, True)
    mask = torch.where(any_gate, gate, fallback_mask)
    return torch.where(mask, base, torch.full_like(base, -1e9))


def _estimate_confidence_target(
    generated: torch.Tensor,
    oracle_actions: torch.Tensor,
    oracle_quality_mask: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    if oracle_actions.numel() == 0:
        return torch.zeros((generated.shape[0],), device=generated.device, dtype=torch.float32)
    dist = _pairwise_weighted_l1(generated.unsqueeze(0), oracle_actions.unsqueeze(0), scale)[0]
    nearest = torch.argmin(dist, dim=-1)
    return oracle_quality_mask[nearest].float().to(generated.device)


def _row_losses(
    sample: dict[str, torch.Tensor],
    outputs: dict[str, torch.Tensor],
    *,
    scale: torch.Tensor,
    frontier_cover_weight: float,
    best_safe_weight: float,
    best_geom_weight: float,
    yaw_weight: float,
    yaw_mag_weight: float,
    action_norm_weight: float,
    score_weight: float,
    score_pairwise_weight: float,
    score_utility_alpha: float,
    score_pairwise_margin: float,
    score_safe_pairwise_weight: float,
    score_yaw_pairwise_weight: float,
    score_ce_weight: float,
    score_listwise_weight: float,
    score_target_temperature: float,
    score_safe_bonus: float,
    score_pareto_bonus: float,
    score_safe_risk_budget: float,
    score_yaw_bonus: float,
    yaw_presence_threshold: float,
    yaw_match_tol: float,
    score_supervision_mode: str,
    mode_weight: float,
    diversity_weight: float,
    scale_weight: float,
    train_score_only: bool,
    use_cached_proposals: bool,
    train_multi_head_score: bool,
    multi_best_safe_ce_weight: float,
    multi_pareto_bce_weight: float,
    multi_yaw_bce_weight: float,
    multi_risk_safe_bce_weight: float,
    multi_geometry_reg_weight: float,
    multi_utility_rank_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    proposals = outputs["proposal_actions_local"]
    scores = outputs["proposal_scores"]
    mode_logits = outputs["mode_logits"]

    oracle_actions = sample["proposal_actions_local"].to(proposals.device)
    geom_top1 = oracle_actions[sample["proposal_geom_top1_index"].long().to(proposals.device)]
    best_safe = oracle_actions[sample["proposal_best_safe_index"].long().to(proposals.device)]
    frontier = _build_frontier_targets(sample).to(proposals.device)
    frontier_mask = _safe_target_mask(sample)
    quality_mask = _quality_targets(sample).to(proposals.device)

    pair_frontier = _pairwise_weighted_l1(proposals.unsqueeze(0), frontier.unsqueeze(0), scale)[0]
    cover_loss = torch.min(pair_frontier, dim=0).values.mean() if frontier.shape[0] > 0 else torch.tensor(0.0, device=proposals.device)

    safe_loss = torch.min(_weighted_l1(proposals, best_safe.unsqueeze(0), scale), dim=0).values
    geom_loss = torch.min(_weighted_l1(proposals, geom_top1.unsqueeze(0), scale), dim=0).values
    target_yaw = torch.tensor(float(sample["proposal_target_delta_local"][5].item()), device=proposals.device, dtype=proposals.dtype)
    if abs(float(target_yaw.item())) > 0.02:
        yaw_loss = torch.min(torch.abs(proposals[:, 5] - target_yaw) / torch.clamp(scale[5].to(proposals.device, proposals.dtype), min=1e-6), dim=0).values
    else:
        yaw_loss = torch.tensor(0.0, device=proposals.device, dtype=proposals.dtype)
    target_yaw_mag = torch.abs(target_yaw)
    yaw_mag_loss = torch.min(
        torch.abs(torch.abs(proposals[:, 5]) - target_yaw_mag) / torch.clamp(scale[5].to(proposals.device, proposals.dtype), min=1e-6),
        dim=0,
    ).values
    target_norm = torch.linalg.norm(best_safe.unsqueeze(0) / torch.clamp(scale.to(proposals.device), min=1e-6), dim=-1)
    pred_norm = torch.linalg.norm(proposals / torch.clamp(scale.to(proposals.device), min=1e-6), dim=-1)
    action_norm_loss = torch.min(torch.abs(pred_norm - target_norm), dim=0).values

    pair_oracle = _pairwise_weighted_l1(proposals.unsqueeze(0), oracle_actions.unsqueeze(0), scale)[0]
    nearest = torch.argmin(pair_oracle, dim=-1)
    conf_target = quality_mask[nearest].float().to(proposals.device)
    score_bce_loss = F.binary_cross_entropy_with_logits(scores, conf_target)
    score_ce_loss = torch.tensor(0.0, device=proposals.device, dtype=proposals.dtype)
    score_listwise_loss = torch.tensor(0.0, device=proposals.device, dtype=proposals.dtype)
    score_pairwise_loss = torch.tensor(0.0, device=proposals.device, dtype=proposals.dtype)
    score_safe_pairwise_loss = torch.tensor(0.0, device=proposals.device, dtype=proposals.dtype)
    score_yaw_pairwise_loss = torch.tensor(0.0, device=proposals.device, dtype=proposals.dtype)

    if str(score_supervision_mode).lower() == "generated":
        current_pose = sample.get("privileged_current_pose_7d", sample.get("current_pose_7d"))
        target_pose = sample.get("privileged_motion_target_pose_7d", sample.get("privileged_basin_center_pose_7d", sample.get("target_pose_7d")))
        if current_pose is None or target_pose is None:
            raise KeyError("generated score supervision requires current_pose_7d and target_pose_7d (or privileged variants)")
        eval_out = evaluate_state_conditioned_proposals(
            current_pose=current_pose.detach().cpu().numpy()[None, :],
            target_pose=target_pose.detach().cpu().numpy()[None, :],
            candidate_actions=proposals.detach().cpu().numpy()[None, :, :],
            contact=np.asarray([float(sample.get("contact_label", torch.tensor(0.0)).item())], dtype=np.float32),
            force_spike=np.asarray([float(sample.get("force_spike_label", torch.tensor(0.0)).item())], dtype=np.float32),
            jam=np.asarray([float(sample.get("jam_label", torch.tensor(0.0)).item())], dtype=np.float32),
            motion_stall=np.asarray([float(sample.get("motion_stall_label", torch.tensor(0.0)).item())], dtype=np.float32),
            near_depth=np.asarray([float(sample.get("near_depth_label", torch.tensor(0.0)).item())], dtype=np.float32),
            kin_invalid=np.asarray([float(sample.get("kinematic_invalid_label", torch.tensor(0.0)).item())], dtype=np.float32),
            action_range_invalid=np.asarray([float(sample.get("action_range_invalid_label", torch.tensor(0.0)).item())], dtype=np.float32),
            gripper_state=np.asarray([float(sample.get("gripper_state", torch.tensor(0.0)).item())], dtype=np.float32),
        )
        cand_geom = torch.as_tensor(eval_out["candidate_geometry_cost"][0], device=proposals.device, dtype=proposals.dtype)
        cand_risk = torch.as_tensor(eval_out["candidate_risk_cost"][0], device=proposals.device, dtype=proposals.dtype)
        base_geom = float(sample["proposal_geometry_cost"][sample["proposal_baseline_index"]].item())
        base_risk = float(sample["proposal_risk_cost"][sample["proposal_baseline_index"]].item())
        cand_gain = base_geom - cand_geom
        cand_risk_delta = cand_risk - base_risk
        pareto_mask = _pareto_mask_from_gain_risk(cand_gain, cand_risk_delta)
        safe_mask = (cand_gain > 0.0) & (cand_risk_delta <= float(score_safe_risk_budget))
        utility = cand_gain - float(score_utility_alpha) * torch.clamp(cand_risk_delta, min=0.0)
        safe_utility = cand_gain - torch.clamp(cand_risk_delta, min=0.0)
        target_yaw = float(sample["proposal_target_delta_local"][5].item())
        yaw_correct = torch.zeros_like(cand_gain, dtype=torch.float32)
        if abs(target_yaw) > 0.02:
            yaw = proposals[:, 5]
            yaw_correct = (((yaw > 0) & (target_yaw > 0)) | ((yaw < 0) & (target_yaw < 0))).to(proposals.dtype)
        yaw_utility = utility + float(score_yaw_bonus) * yaw_correct
        score_bce_loss = F.binary_cross_entropy_with_logits(scores, torch.maximum(pareto_mask.float(), safe_mask.float()))
        score_pairwise_loss = _pairwise_rank_loss(scores, utility, margin=score_pairwise_margin)
        score_safe_pairwise_loss = _pairwise_rank_loss(scores, safe_utility, margin=score_pairwise_margin)
        score_yaw_pairwise_loss = _pairwise_rank_loss(scores, yaw_utility, margin=score_pairwise_margin)
        if str(score_supervision_mode).lower() == "generated_rank":
            target_util = utility.clone()
            if torch.any(safe_mask):
                target_util = torch.where(
                    safe_mask,
                    safe_utility + float(score_yaw_bonus) * yaw_correct,
                    torch.full_like(target_util, -1e9),
                )
            elif torch.any(pareto_mask):
                target_util = torch.where(
                    pareto_mask,
                    utility + float(score_yaw_bonus) * yaw_correct,
                    torch.full_like(target_util, -1e9),
                )
            else:
                target_util = utility + float(score_yaw_bonus) * yaw_correct
            target_idx = torch.argmax(target_util).unsqueeze(0)
            score_ce_loss = F.cross_entropy(scores.unsqueeze(0), target_idx)
        elif str(score_supervision_mode).lower() == "generated_listwise":
            target_logits = utility.clone()
            target_logits = target_logits + float(score_safe_bonus) * safe_mask.float()
            target_logits = target_logits + float(score_pareto_bonus) * pareto_mask.float()
            target_logits = target_logits + float(score_yaw_bonus) * yaw_correct
            target_idx = torch.argmax(target_logits).unsqueeze(0)
            temp = max(float(score_target_temperature), 1e-3)
            target_probs = F.softmax(target_logits / temp, dim=0).detach()
            score_ce_loss = F.cross_entropy(scores.unsqueeze(0), target_idx)
            score_listwise_loss = F.kl_div(
                F.log_softmax(scores / temp, dim=0),
                target_probs,
                reduction="batchmean",
            )
            score_pairwise_loss = _pairwise_rank_loss(scores, target_logits, margin=score_pairwise_margin)
            score_safe_pairwise_loss = _pairwise_rank_loss(scores, safe_utility + float(score_safe_bonus) * safe_mask.float(), margin=score_pairwise_margin)
            score_yaw_pairwise_loss = _pairwise_rank_loss(scores, utility + float(score_yaw_bonus) * yaw_correct, margin=score_pairwise_margin)
    else:
        utility_target = sample["proposal_geometry_gain"].to(proposals.device) - float(score_utility_alpha) * torch.clamp(
            sample["proposal_risk_delta"].to(proposals.device), min=0.0
        )
        utility_nearest = utility_target[nearest]
        score_pairwise_loss = _pairwise_rank_loss(scores, utility_nearest, margin=score_pairwise_margin)

    score_loss = (
        score_bce_loss
        + float(score_ce_weight) * score_ce_loss
        + float(score_listwise_weight) * score_listwise_loss
        + float(score_pairwise_weight) * score_pairwise_loss
        + float(score_safe_pairwise_weight) * score_safe_pairwise_loss
        + float(score_yaw_pairwise_weight) * score_yaw_pairwise_loss
    )
    multi_head_loss = torch.tensor(0.0, device=proposals.device, dtype=proposals.dtype)
    multi_stats: dict[str, float] = {}
    if train_multi_head_score:
        multi_head_loss, multi_stats = _multi_head_score_loss(
            sample,
            outputs,
            score_utility_alpha=score_utility_alpha,
            score_safe_risk_budget=score_safe_risk_budget,
            yaw_presence_threshold=yaw_presence_threshold,
            yaw_match_tol=yaw_match_tol,
            best_safe_ce_weight=multi_best_safe_ce_weight,
            pareto_bce_weight=multi_pareto_bce_weight,
            yaw_bce_weight=multi_yaw_bce_weight,
            risk_safe_bce_weight=multi_risk_safe_bce_weight,
            geometry_reg_weight=multi_geometry_reg_weight,
            utility_rank_weight=multi_utility_rank_weight,
        )

    dist_proposals = torch.pdist(proposals / torch.clamp(scale.to(proposals.device), min=1e-6), p=2)
    diversity_loss = torch.relu(0.50 - dist_proposals).mean() if dist_proposals.numel() > 0 else torch.tensor(0.0, device=proposals.device)
    scale_loss = torch.mean(torch.relu(torch.abs(proposals) - (1.5 * scale.to(proposals.device))).pow(2))

    mode_id = _mode_target(sample).to(mode_logits.device)
    if int(mode_id.item()) >= 0 and int(mode_id.item()) < mode_logits.shape[-1]:
        mode_loss = F.cross_entropy(mode_logits.unsqueeze(0), mode_id.unsqueeze(0))
        mode_acc = float(torch.argmax(mode_logits).item() == int(mode_id.item()))
    else:
        mode_loss = torch.tensor(0.0, device=proposals.device)
        mode_acc = 0.0

    if train_score_only:
        if train_multi_head_score:
            total = multi_head_loss
        else:
            total = float(score_weight) * score_loss
    else:
        total = (
            float(frontier_cover_weight) * cover_loss
            + float(best_safe_weight) * safe_loss
            + float(best_geom_weight) * geom_loss
            + float(yaw_weight) * yaw_loss
            + float(yaw_mag_weight) * yaw_mag_loss
            + float(action_norm_weight) * action_norm_loss
            + float(score_weight) * score_loss
            + float(mode_weight) * mode_loss
            + float(diversity_weight) * diversity_loss
            + float(scale_weight) * scale_loss
        )
    stats = {
        "cover_loss": float(cover_loss.detach().cpu().item()),
        "best_safe_loss": float(safe_loss.detach().cpu().item()),
        "best_geom_loss": float(geom_loss.detach().cpu().item()),
        "yaw_loss": float(yaw_loss.detach().cpu().item()),
        "yaw_mag_loss": float(yaw_mag_loss.detach().cpu().item()),
        "action_norm_loss": float(action_norm_loss.detach().cpu().item()),
        "score_loss": float(score_loss.detach().cpu().item()),
        "score_bce_loss": float(score_bce_loss.detach().cpu().item()),
        "score_ce_loss": float(score_ce_loss.detach().cpu().item()),
        "score_listwise_loss": float(score_listwise_loss.detach().cpu().item()),
        "score_pairwise_loss": float(score_pairwise_loss.detach().cpu().item()),
        "multi_head_loss": float(multi_head_loss.detach().cpu().item()),
        "mode_loss": float(mode_loss.detach().cpu().item()),
        "diversity_loss": float(diversity_loss.detach().cpu().item()),
        "scale_loss": float(scale_loss.detach().cpu().item()),
        "mode_acc": mode_acc,
        "frontier_size": float(frontier.shape[0]),
        "quality_positive_rate": float(frontier_mask.float().mean().item()),
        "best_safe_recall": float(
            torch.any(_weighted_l1(proposals, best_safe.unsqueeze(0), scale) <= 0.75, dim=0).float().item()
        ),
        "best_geom_recall": float(
            torch.any(_weighted_l1(proposals, geom_top1.unsqueeze(0), scale) <= 0.75, dim=0).float().item()
        ),
    }
    stats.update(multi_stats)
    return total, stats


@torch.no_grad()
def _evaluate_split(
    model: DepthForceLocalProposalPolicy,
    loader: DataLoader,
    dataset: DepthForceLocalProposalDataset,
    *,
    device: torch.device,
    scale: torch.Tensor,
    score_pairwise_weight: float,
    score_utility_alpha: float,
    score_pairwise_margin: float,
    score_safe_pairwise_weight: float,
    score_yaw_pairwise_weight: float,
    score_ce_weight: float,
    score_listwise_weight: float,
    score_target_temperature: float,
    score_safe_bonus: float,
    score_pareto_bonus: float,
    score_safe_risk_budget: float,
    score_yaw_bonus: float,
    yaw_presence_threshold: float,
    yaw_match_tol: float,
    score_supervision_mode: str,
    train_score_only: bool,
    use_cached_proposals: bool,
    use_multi_head_selection: bool,
    multi_head_selection_mode: str,
    train_multi_head_score: bool,
    multi_best_safe_ce_weight: float,
    multi_pareto_bce_weight: float,
    multi_yaw_bce_weight: float,
    multi_risk_safe_bce_weight: float,
    multi_geometry_reg_weight: float,
    multi_utility_rank_weight: float,
    multi_utility_w_safe: float,
    multi_utility_w_pareto: float,
    multi_utility_w_yaw: float,
    multi_utility_w_geom: float,
    multi_utility_w_risk: float,
) -> dict[str, float]:
    model.eval()
    totals = {
        "rows": 0,
        "cover_loss": 0.0,
        "best_safe_loss": 0.0,
        "best_geom_loss": 0.0,
        "yaw_loss": 0.0,
        "yaw_mag_loss": 0.0,
        "action_norm_loss": 0.0,
        "score_loss": 0.0,
        "score_ce_loss": 0.0,
        "mode_loss": 0.0,
        "diversity_loss": 0.0,
        "scale_loss": 0.0,
        "multi_head_loss": 0.0,
        "mode_acc": 0.0,
        "frontier_size": 0.0,
        "quality_positive_rate": 0.0,
        "best_safe_recall": 0.0,
        "best_geom_recall": 0.0,
        "selected_best_safe_hit": 0.0,
        "selected_best_geom_hit": 0.0,
        "selected_pareto_hit": 0.0,
        "selected_geometry_gain_mean": 0.0,
        "selected_risk_delta_mean": 0.0,
        "selected_yaw_presence_rate": 0.0,
        "selected_yaw_match_rate": 0.0,
        "selected_correct_yaw_sign_rate": 0.0,
        "oracle_best_geometry_gain_mean": 0.0,
        "oracle_best_risk_delta_mean": 0.0,
        "oracle_best_yaw_presence_rate": 0.0,
        "oracle_best_yaw_match_rate": 0.0,
        "oracle_best_correct_yaw_sign_rate": 0.0,
    }
    eps = 1e-6
    for batch in loader:
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        outputs = model(
            front_rgb=batch["front_rgb"],
            wrist_rgb=batch["wrist_rgb"],
            wrist_depth=batch["wrist_depth"],
            force_history=batch["force_history"],
            proprio=batch["proprio"],
            planner_base_action_local=batch["planner_base_action_local"],
            proposal_actions_local=batch["proposal_actions_local"] if use_cached_proposals else None,
            stage_token=batch.get("stage_token"),
            contact_phase=batch.get("contact_phase"),
            depth_proximity=batch.get("depth_proximity"),
            gripper_state=batch.get("gripper_state"),
        )
        proposals = outputs["proposal_actions_local"]
        scores = (
            _multi_head_layered_score(
                outputs,
                mode=multi_head_selection_mode,
                w_safe=multi_utility_w_safe,
                w_pareto=multi_utility_w_pareto,
                w_yaw=multi_utility_w_yaw,
                w_geom=multi_utility_w_geom,
                w_risk=multi_utility_w_risk,
            )
            if use_multi_head_selection
            else outputs["proposal_scores"]
        )
        oracle_actions = batch["proposal_actions_local"]
        row_indices = batch["row_index"].detach().cpu().numpy().astype(np.int64)
        geom_top1 = oracle_actions[torch.arange(oracle_actions.shape[0], device=device), batch["proposal_geom_top1_index"].long()]
        best_safe = oracle_actions[torch.arange(oracle_actions.shape[0], device=device), batch["proposal_best_safe_index"].long()]
        frontier_mask = _safe_target_mask(batch)
        frontier = []
        for i in range(oracle_actions.shape[0]):
            front = oracle_actions[i][frontier_mask[i]]
            if front.numel() == 0:
                front = best_safe[i].unsqueeze(0)
            frontier.append(front)

        cover_loss = 0.0
        safe_loss = 0.0
        geom_loss = 0.0
        yaw_loss = 0.0
        score_loss = 0.0
        score_ce_loss = 0.0
        mode_loss = 0.0
        diversity_loss = 0.0
        scale_loss = 0.0
        multi_head_loss = 0.0
        mode_acc = 0.0
        best_safe_recall = 0.0
        best_geom_recall = 0.0
        selected_best_safe_hit = 0.0
        selected_best_geom_hit = 0.0
        selected_pareto_hit = 0.0
        selected_geometry_gain = 0.0
        selected_risk_delta = 0.0
        selected_yaw_presence_rate = 0.0
        selected_yaw_match_rate = 0.0
        selected_correct_yaw_sign = 0.0
        oracle_best_geometry_gain = 0.0
        oracle_best_risk_delta = 0.0
        oracle_best_yaw_presence_rate = 0.0
        oracle_best_yaw_match_rate = 0.0
        oracle_best_correct_yaw_sign = 0.0

        for i in range(proposals.shape[0]):
            row_idx = int(row_indices[i])
            base_geom = float(dataset.data["proposal_geometry_cost"][row_idx, dataset.data["proposal_baseline_index"][row_idx]])
            base_risk = float(dataset.data["proposal_risk_cost"][row_idx, dataset.data["proposal_baseline_index"][row_idx]])
            row = {
                "proposal_actions_local": oracle_actions[i].detach().cpu(),
                "proposal_pareto_mask": batch["proposal_pareto_mask"][i].detach().cpu(),
                "proposal_budget_mask": batch["proposal_budget_mask"][i].detach().cpu(),
                "proposal_geometry_gain": batch["proposal_geometry_gain"][i].detach().cpu(),
                "proposal_risk_delta": batch["proposal_risk_delta"][i].detach().cpu(),
                "proposal_geom_top1_index": batch["proposal_geom_top1_index"][i].detach().cpu(),
                "proposal_best_safe_index": batch["proposal_best_safe_index"][i].detach().cpu(),
                "proposal_target_mode_id": batch["proposal_target_mode_id"][i].detach().cpu(),
                "proposal_target_delta_local": batch["proposal_target_delta_local"][i].detach().cpu(),
            }
            total_loss, stats = _row_losses(
                row,
                {
                    "proposal_actions_local": proposals[i],
                    "proposal_scores": scores[i],
                    "multi_head_scores": outputs["multi_head_scores"][i],
                    "mode_logits": outputs["mode_logits"][i],
                },
                scale=scale,
                frontier_cover_weight=1.0,
                best_safe_weight=1.0,
                best_geom_weight=0.5,
                yaw_weight=1.0,
                yaw_mag_weight=0.0,
                action_norm_weight=0.0,
                score_weight=0.25,
                score_pairwise_weight=score_pairwise_weight,
                score_utility_alpha=score_utility_alpha,
                score_pairwise_margin=score_pairwise_margin,
                score_safe_pairwise_weight=0.0,
                score_yaw_pairwise_weight=0.0,
                score_ce_weight=0.0,
                score_listwise_weight=0.0,
                score_target_temperature=0.5,
                score_safe_bonus=1.0,
                score_pareto_bonus=0.5,
                score_safe_risk_budget=0.03,
                score_yaw_bonus=0.25,
                yaw_presence_threshold=0.0025,
                yaw_match_tol=0.0015,
                score_supervision_mode="oracle_bank",
                mode_weight=0.25,
                diversity_weight=0.05,
                scale_weight=0.01,
                train_score_only=train_score_only,
                use_cached_proposals=use_cached_proposals,
                train_multi_head_score=train_multi_head_score,
                multi_best_safe_ce_weight=multi_best_safe_ce_weight,
                multi_pareto_bce_weight=multi_pareto_bce_weight,
                multi_yaw_bce_weight=multi_yaw_bce_weight,
                multi_risk_safe_bce_weight=multi_risk_safe_bce_weight,
                multi_geometry_reg_weight=multi_geometry_reg_weight,
                multi_utility_rank_weight=multi_utility_rank_weight,
            )
            cover_loss += stats["cover_loss"]
            safe_loss += stats["best_safe_loss"]
            geom_loss += stats["best_geom_loss"]
            yaw_loss += stats["yaw_loss"]
            score_loss += stats["score_loss"]
            score_ce_loss += stats["score_ce_loss"]
            mode_loss += stats["mode_loss"]
            diversity_loss += stats["diversity_loss"]
            scale_loss += stats["scale_loss"]
            multi_head_loss += stats.get("multi_head_loss", 0.0)
            mode_acc += stats["mode_acc"]
            best_safe_recall += stats["best_safe_recall"]
            best_geom_recall += stats["best_geom_recall"]

            selected_idx = int(torch.argmax(scores[i]).item())
            sel = proposals[i, selected_idx]
            sel_dist_safe = _weighted_l1(sel.unsqueeze(0), best_safe[i].unsqueeze(0), scale)[0].item()
            sel_dist_geom = _weighted_l1(sel.unsqueeze(0), geom_top1[i].unsqueeze(0), scale)[0].item()
            selected_best_safe_hit += float(sel_dist_safe <= 0.75)
            selected_best_geom_hit += float(sel_dist_geom <= 0.75)

            if frontier[i].numel() > 0:
                sel_dist_frontier = _pairwise_weighted_l1(sel.unsqueeze(0).unsqueeze(0), frontier[i].unsqueeze(0), scale)[0, 0]
                selected_pareto_hit += float(torch.min(sel_dist_frontier).item() <= 0.75)
            else:
                selected_pareto_hit += 0.0

            from local_proposal_utils import evaluate_state_conditioned_proposals

            eval_out = evaluate_state_conditioned_proposals(
                current_pose=np.asarray(batch["current_pose_7d"][i].detach().cpu().numpy(), dtype=np.float32)[None, :],
                target_pose=np.asarray(batch["target_pose_7d"][i].detach().cpu().numpy(), dtype=np.float32)[None, :],
                candidate_actions=sel.detach().cpu().numpy()[None, None, :],
                contact=np.asarray([float(dataset.data.get("contact_label", np.zeros((len(dataset),), dtype=np.float32))[row_idx])], dtype=np.float32),
                force_spike=np.asarray([float(dataset.data.get("force_spike_label", np.zeros((len(dataset),), dtype=np.float32))[row_idx])], dtype=np.float32),
                jam=np.asarray([float(dataset.data.get("jam_label", np.zeros((len(dataset),), dtype=np.float32))[row_idx])], dtype=np.float32),
                motion_stall=np.asarray([float(dataset.data.get("motion_stall_label", np.zeros((len(dataset),), dtype=np.float32))[row_idx])], dtype=np.float32),
                near_depth=np.asarray([float(dataset.data.get("near_depth_label", np.zeros((len(dataset),), dtype=np.float32))[row_idx])], dtype=np.float32),
                kin_invalid=np.asarray([float(dataset.data.get("kinematic_invalid_label", np.zeros((len(dataset),), dtype=np.float32))[row_idx])], dtype=np.float32),
                action_range_invalid=np.asarray([float(dataset.data.get("action_range_invalid_label", np.zeros((len(dataset),), dtype=np.float32))[row_idx])], dtype=np.float32),
                gripper_state=np.asarray([float(batch["gripper_state"][i].detach().cpu().item())], dtype=np.float32),
            )
            sel_geom_eval = float(eval_out["candidate_geometry_cost"][0, 0])
            sel_risk_eval = float(eval_out["candidate_risk_cost"][0, 0])
            selected_geometry_gain += float(base_geom - sel_geom_eval)
            selected_risk_delta += float(sel_risk_eval - base_risk)
            sel_yaw_abs = float(abs(sel[5].item()))
            best_safe_yaw_abs = float(abs(best_safe[i, 5].item()))
            selected_yaw_presence_rate += float(sel_yaw_abs > float(yaw_presence_threshold))
            selected_yaw_match_rate += float(abs(sel_yaw_abs - best_safe_yaw_abs) <= float(yaw_match_tol))
            tgt_yaw = float(batch["proposal_target_delta_local"][i, 5].item())
            if abs(tgt_yaw) > float(yaw_presence_threshold):
                selected_correct_yaw_sign += float((sel[5].item() > 0 and tgt_yaw > 0) or (sel[5].item() < 0 and tgt_yaw < 0))

            oracle_utils = batch["proposal_geometry_gain"][i] - torch.clamp(batch["proposal_risk_delta"][i], min=0.0)
            best_oracle_idx = int(torch.argmax(oracle_utils).item())
            oracle_best_geometry_gain += float(batch["proposal_geometry_gain"][i][best_oracle_idx].item())
            oracle_best_risk_delta += float(batch["proposal_risk_delta"][i][best_oracle_idx].item())
            oracle_best_yaw_abs = float(abs(oracle_actions[i, best_oracle_idx, 5].item()))
            oracle_best_yaw_presence_rate += float(oracle_best_yaw_abs > float(yaw_presence_threshold))
            oracle_best_yaw_match_rate += float(abs(oracle_best_yaw_abs - best_safe_yaw_abs) <= float(yaw_match_tol))
            if abs(tgt_yaw) > float(yaw_presence_threshold):
                oracle_best_correct_yaw_sign += float(
                    (oracle_actions[i, best_oracle_idx, 5].item() > 0 and tgt_yaw > 0)
                    or (oracle_actions[i, best_oracle_idx, 5].item() < 0 and tgt_yaw < 0)
                )

        rows = float(proposals.shape[0])
        totals["rows"] += int(rows)
        totals["cover_loss"] += cover_loss / rows
        totals["best_safe_loss"] += safe_loss / rows
        totals["best_geom_loss"] += geom_loss / rows
        totals["yaw_loss"] += yaw_loss / rows
        totals["score_loss"] += score_loss / rows
        totals["score_ce_loss"] += score_ce_loss / rows
        totals["mode_loss"] += mode_loss / rows
        totals["diversity_loss"] += diversity_loss / rows
        totals["scale_loss"] += scale_loss / rows
        totals["multi_head_loss"] += multi_head_loss / rows
        totals["mode_acc"] += mode_acc / rows
        totals["frontier_size"] += float(np.mean([frontier[i].shape[0] for i in range(len(frontier))]))
        totals["quality_positive_rate"] += float(frontier_mask.float().mean().item())
        totals["best_safe_recall"] += best_safe_recall / rows
        totals["best_geom_recall"] += best_geom_recall / rows
        totals["selected_best_safe_hit"] += selected_best_safe_hit / rows
        totals["selected_best_geom_hit"] += selected_best_geom_hit / rows
        totals["selected_pareto_hit"] += selected_pareto_hit / rows
        totals["selected_geometry_gain_mean"] += selected_geometry_gain / rows
        totals["selected_risk_delta_mean"] += selected_risk_delta / rows
        totals["selected_yaw_presence_rate"] += selected_yaw_presence_rate / rows
        totals["selected_yaw_match_rate"] += selected_yaw_match_rate / rows
        totals["selected_correct_yaw_sign_rate"] += selected_correct_yaw_sign / rows
        totals["oracle_best_geometry_gain_mean"] += oracle_best_geometry_gain / rows
        totals["oracle_best_risk_delta_mean"] += oracle_best_risk_delta / rows
        totals["oracle_best_yaw_presence_rate"] += oracle_best_yaw_presence_rate / rows
        totals["oracle_best_yaw_match_rate"] += oracle_best_yaw_match_rate / rows
        totals["oracle_best_correct_yaw_sign_rate"] += oracle_best_correct_yaw_sign / rows

    denom = max(len(loader), 1)
    return {k: float(v / denom) if isinstance(v, float) else v for k, v in totals.items()}


def _train_fold(
    dataset: DepthForceLocalProposalDataset,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    args,
    *,
    fold_name: str,
    output_dir: Path,
    model_kwargs: dict[str, bool],
    init_ckpt: dict | None,
) -> dict:
    device = torch.device(args.device)
    load_summary: dict[str, list[str]] | None = None
    if init_ckpt is not None:
        ckpt_model_kwargs = dict(init_ckpt.get("model_kwargs", {}))
        merged_kwargs = {**ckpt_model_kwargs, **model_kwargs}
        model = DepthForceLocalProposalPolicy(
            proposal_count=int(init_ckpt.get("proposal_count", args.proposal_count)),
            hidden_dim=int(init_ckpt.get("hidden_dim", args.hidden_dim)),
            state_dim=int(init_ckpt.get("state_dim", args.state_dim)),
            **merged_kwargs,
        )
        missing, unexpected = model.load_state_dict(init_ckpt["model_state_dict"], strict=False)
        load_summary = _summarize_state_dict_load(model, init_ckpt["model_state_dict"])
        print(
            "[load] "
            f"missing={load_summary['missing_keys']} "
            f"unexpected={load_summary['unexpected_keys']} "
            f"loaded={len(load_summary['loaded_keys'])} "
            f"newly_initialized={load_summary['newly_initialized_keys']}",
            flush=True,
        )
        if missing or unexpected:
            print(f"[load/raw] missing={missing} unexpected={unexpected}", flush=True)
    else:
        model = DepthForceLocalProposalPolicy(
            proposal_count=args.proposal_count,
            hidden_dim=args.hidden_dim,
            state_dim=args.state_dim,
            **model_kwargs,
        )
    if args.train_score_only:
        _freeze_score_only(model, observability_tune=bool(args.score_observability_tune))
    model = model.to(device)
    trainable_names = _validate_trainable_parameters(model)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        raise RuntimeError("no trainable parameters remain after applying freeze settings")
    print(
        "[trainable] "
        f"count={len(trainable_names)} "
        f"names={trainable_names}",
        flush=True,
    )
    opt = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    train_loader = DataLoader(Subset(dataset, train_idx.tolist()), batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(Subset(dataset, val_idx.tolist()), batch_size=args.batch_size, shuffle=False, num_workers=0)
    scale = ACTION_SCALE.to(device)
    best_state = None
    best_metric = -1e9
    history: list[dict[str, float]] = []

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        epoch_stats = {
            "cover_loss": 0.0,
            "best_safe_loss": 0.0,
            "best_geom_loss": 0.0,
            "yaw_loss": 0.0,
            "yaw_mag_loss": 0.0,
            "action_norm_loss": 0.0,
            "score_loss": 0.0,
            "mode_loss": 0.0,
            "diversity_loss": 0.0,
            "scale_loss": 0.0,
            "multi_head_loss": 0.0,
            "mode_acc": 0.0,
        }
        num_batches = 0
        for batch in train_loader:
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            outputs = model(
                front_rgb=batch["front_rgb"],
                wrist_rgb=batch["wrist_rgb"],
                wrist_depth=batch["wrist_depth"],
                force_history=batch["force_history"],
                proprio=batch["proprio"],
                planner_base_action_local=batch["planner_base_action_local"],
                proposal_actions_local=batch["proposal_actions_local"] if args.proposal_cache_npz else None,
                stage_token=batch.get("stage_token"),
                contact_phase=batch.get("contact_phase"),
                depth_proximity=batch.get("depth_proximity"),
                gripper_state=batch.get("gripper_state"),
            )
            batch_losses = []
            row_stat_sums = {k: 0.0 for k in epoch_stats}
            for i in range(int(batch["front_rgb"].shape[0])):
                row = {k: v[i].detach().cpu() if torch.is_tensor(v) and v.ndim > 0 else v for k, v in batch.items()}
                loss, stats = _row_losses(
                row,
                {
                    "proposal_actions_local": outputs["proposal_actions_local"][i],
                    "proposal_scores": outputs["proposal_scores"][i],
                    "multi_head_scores": outputs["multi_head_scores"][i],
                    "mode_logits": outputs["mode_logits"][i],
                },
                    scale=scale,
                    frontier_cover_weight=args.frontier_cover_weight,
                    best_safe_weight=args.best_safe_weight,
                    best_geom_weight=args.best_geom_weight,
                    yaw_weight=args.yaw_weight,
                    yaw_mag_weight=args.yaw_mag_weight,
                    action_norm_weight=args.action_norm_weight,
                    score_weight=args.score_weight,
                    score_pairwise_weight=args.score_pairwise_weight,
                    score_utility_alpha=args.score_utility_alpha,
                    score_pairwise_margin=args.score_pairwise_margin,
                    score_safe_pairwise_weight=args.score_safe_pairwise_weight,
                    score_yaw_pairwise_weight=args.score_yaw_pairwise_weight,
                    score_ce_weight=args.score_ce_weight,
                    score_listwise_weight=args.score_listwise_weight,
                    score_target_temperature=args.score_target_temperature,
                    score_safe_bonus=args.score_safe_bonus,
                    score_pareto_bonus=args.score_pareto_bonus,
                    score_safe_risk_budget=args.score_safe_risk_budget,
                    score_yaw_bonus=args.score_yaw_bonus,
                    yaw_presence_threshold=args.yaw_presence_threshold,
                    yaw_match_tol=args.yaw_match_tol,
                    score_supervision_mode=args.score_supervision_mode,
                    mode_weight=args.mode_weight,
                    diversity_weight=args.diversity_weight,
                    scale_weight=args.scale_reg_weight,
                    train_score_only=bool(args.train_score_only),
                    use_cached_proposals=bool(args.proposal_cache_npz),
                    train_multi_head_score=bool(args.train_multi_head_score),
                    multi_best_safe_ce_weight=args.multi_best_safe_ce_weight,
                    multi_pareto_bce_weight=args.multi_pareto_bce_weight,
                    multi_yaw_bce_weight=args.multi_yaw_bce_weight,
                    multi_risk_safe_bce_weight=args.multi_risk_safe_bce_weight,
                    multi_geometry_reg_weight=args.multi_geometry_reg_weight,
                    multi_utility_rank_weight=args.multi_utility_rank_weight,
                )
                batch_losses.append(loss)
                for k in row_stat_sums:
                    if k in stats:
                        row_stat_sums[k] += float(stats[k])
            loss = torch.stack(batch_losses).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            epoch_loss += float(loss.detach().cpu().item())
            for k in epoch_stats:
                epoch_stats[k] += row_stat_sums[k] / max(int(batch["front_rgb"].shape[0]), 1)
            num_batches += 1
        train_summary = {k: float(v / max(num_batches, 1)) for k, v in epoch_stats.items()}
        train_summary["loss"] = float(epoch_loss / max(num_batches, 1))
        val_report = _evaluate_split(
            model,
            val_loader,
            dataset,
            device=device,
            scale=scale,
            score_pairwise_weight=args.score_pairwise_weight,
            score_utility_alpha=args.score_utility_alpha,
            score_pairwise_margin=args.score_pairwise_margin,
            score_safe_pairwise_weight=args.score_safe_pairwise_weight,
            score_yaw_pairwise_weight=args.score_yaw_pairwise_weight,
            score_ce_weight=args.score_ce_weight,
            score_listwise_weight=args.score_listwise_weight,
            score_target_temperature=args.score_target_temperature,
            score_safe_bonus=args.score_safe_bonus,
            score_pareto_bonus=args.score_pareto_bonus,
            score_safe_risk_budget=args.score_safe_risk_budget,
            score_yaw_bonus=args.score_yaw_bonus,
            yaw_presence_threshold=args.yaw_presence_threshold,
            yaw_match_tol=args.yaw_match_tol,
            score_supervision_mode=args.score_supervision_mode,
            train_score_only=bool(args.train_score_only),
            use_cached_proposals=bool(args.proposal_cache_npz),
            use_multi_head_selection=bool(args.train_multi_head_score),
            multi_head_selection_mode=str(args.multi_head_selection_mode),
            train_multi_head_score=bool(args.train_multi_head_score),
            multi_best_safe_ce_weight=args.multi_best_safe_ce_weight,
            multi_pareto_bce_weight=args.multi_pareto_bce_weight,
            multi_yaw_bce_weight=args.multi_yaw_bce_weight,
            multi_risk_safe_bce_weight=args.multi_risk_safe_bce_weight,
            multi_geometry_reg_weight=args.multi_geometry_reg_weight,
            multi_utility_rank_weight=args.multi_utility_rank_weight,
            multi_utility_w_safe=args.multi_utility_w_safe,
            multi_utility_w_pareto=args.multi_utility_w_pareto,
            multi_utility_w_yaw=args.multi_utility_w_yaw,
            multi_utility_w_geom=args.multi_utility_w_geom,
            multi_utility_w_risk=args.multi_utility_w_risk,
        )
        history.append({"epoch": epoch, "train": train_summary, "val": val_report})
        metric = (
            val_report["selected_best_safe_hit"]
            + val_report["selected_best_geom_hit"]
            + val_report["selected_pareto_hit"]
            + val_report["mode_acc"]
            - val_report["selected_risk_delta_mean"]
        )
        if metric > best_metric:
            best_metric = metric
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    if best_state is None:
        best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"{fold_name}.pt"
    torch.save(
            {
                "model_state_dict": best_state,
                "proposal_count": args.proposal_count,
                "state_dim": args.state_dim,
                "hidden_dim": args.hidden_dim,
                "model_kwargs": model_kwargs,
            },
            ckpt_path,
        )
    return {
        "fold": fold_name,
        "train_rows": int(train_idx.size),
        "val_rows": int(val_idx.size),
        "best_metric": float(best_metric),
        "history": history,
        "checkpoint": str(ckpt_path),
        "final_val": history[-1]["val"] if history else {},
        "init_load_summary": load_summary,
        "trainable_parameter_names": trainable_names,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_npz", required=True)
    ap.add_argument("--proposal_cache_npz", default="")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--proposal_count", type=int, default=8)
    ap.add_argument("--state_dim", type=int, default=384)
    ap.add_argument("--hidden_dim", type=int, default=256)
    ap.add_argument("--init_checkpoint", default=None)
    ap.add_argument("--train_score_only", action="store_true")
    ap.add_argument("--score_observability_tune", action="store_true")
    ap.add_argument("--train_multi_head_score", action="store_true")
    ap.add_argument(
        "--input_mode",
        default="depth_force_proprio_planner",
        choices=("proprio_planner", "force_proprio_planner", "depth_proprio_planner", "depth_force_proprio_planner"),
    )
    ap.add_argument("--frontier_cover_weight", "--cover_weight", dest="frontier_cover_weight", type=float, default=1.0)
    ap.add_argument("--best_safe_weight", type=float, default=1.0)
    ap.add_argument("--best_geom_weight", type=float, default=0.5)
    ap.add_argument("--yaw_weight", type=float, default=1.0)
    ap.add_argument("--yaw_mag_weight", type=float, default=0.0)
    ap.add_argument("--action_norm_weight", type=float, default=0.0)
    ap.add_argument("--score_weight", type=float, default=0.25)
    ap.add_argument("--score_pairwise_weight", type=float, default=0.0)
    ap.add_argument("--score_utility_alpha", type=float, default=0.3)
    ap.add_argument("--score_pairwise_margin", type=float, default=0.05)
    ap.add_argument("--score_safe_pairwise_weight", type=float, default=0.0)
    ap.add_argument("--score_yaw_pairwise_weight", type=float, default=0.0)
    ap.add_argument("--score_ce_weight", type=float, default=0.0)
    ap.add_argument("--score_listwise_weight", type=float, default=0.0)
    ap.add_argument("--score_target_temperature", type=float, default=0.5)
    ap.add_argument("--score_safe_bonus", type=float, default=1.0)
    ap.add_argument("--score_pareto_bonus", type=float, default=0.5)
    ap.add_argument("--score_safe_risk_budget", type=float, default=0.03)
    ap.add_argument("--score_yaw_bonus", type=float, default=0.25)
    ap.add_argument("--multi_best_safe_ce_weight", type=float, default=2.0)
    ap.add_argument("--multi_pareto_bce_weight", type=float, default=1.0)
    ap.add_argument("--multi_yaw_bce_weight", type=float, default=1.0)
    ap.add_argument("--multi_risk_safe_bce_weight", type=float, default=0.5)
    ap.add_argument("--multi_geometry_reg_weight", type=float, default=0.5)
    ap.add_argument("--multi_utility_rank_weight", type=float, default=0.5)
    ap.add_argument("--multi_utility_w_safe", type=float, default=1.0)
    ap.add_argument("--multi_utility_w_pareto", type=float, default=1.0)
    ap.add_argument("--multi_utility_w_yaw", type=float, default=1.0)
    ap.add_argument("--multi_utility_w_geom", type=float, default=0.5)
    ap.add_argument("--multi_utility_w_risk", type=float, default=0.5)
    ap.add_argument(
        "--multi_head_selection_mode",
        default="layered_multi",
        choices=(
            "weighted_multi",
            "layered_multi",
            "pareto_then_best_safe",
            "pareto_then_geometry",
        ),
    )
    ap.add_argument("--yaw_presence_threshold", type=float, default=0.0025)
    ap.add_argument("--yaw_match_tol", type=float, default=0.0015)
    ap.add_argument(
        "--score_supervision_mode",
        default="oracle_bank",
        choices=("oracle_bank", "generated", "generated_rank", "generated_listwise"),
    )
    ap.add_argument("--mode_weight", type=float, default=0.25)
    ap.add_argument("--diversity_weight", type=float, default=0.05)
    ap.add_argument("--scale_reg_weight", "--scale_weight", dest="scale_reg_weight", type=float, default=0.01)
    ap.add_argument("--yaw_scale_factor", type=float, default=1.0)
    ap.add_argument("--rot_scale_factor", type=float, default=1.0)
    ap.add_argument("--xyz_scale_factor", type=float, default=1.0)
    ap.add_argument("--x_scale_factor", type=float, default=1.0)
    ap.add_argument("--y_scale_factor", type=float, default=1.0)
    ap.add_argument("--z_scale_factor", type=float, default=1.0)
    ap.add_argument("--roll_scale_factor", type=float, default=1.0)
    ap.add_argument("--pitch_scale_factor", type=float, default=1.0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset = DepthForceLocalProposalDataset(args.dataset_npz, proposal_cache_npz=args.proposal_cache_npz or None)
    init_ckpt = _load_init_checkpoint(args.init_checkpoint)
    if args.train_score_only and init_ckpt is None:
        raise ValueError("--train_score_only requires --init_checkpoint")
    model_kwargs = _resolve_model_kwargs(args.input_mode)
    axis_scale_changed = any(
        float(v) != 1.0
        for v in (
            args.yaw_scale_factor,
            args.rot_scale_factor,
            args.xyz_scale_factor,
            args.x_scale_factor,
            args.y_scale_factor,
            args.z_scale_factor,
            args.roll_scale_factor,
            args.pitch_scale_factor,
        )
    )
    if axis_scale_changed:
        model_kwargs = dict(model_kwargs)
        model_kwargs["proposal_action_scale"] = LocalProposalActionScale(
            xyz=(
                0.008 * float(args.xyz_scale_factor) * float(args.x_scale_factor),
                0.008 * float(args.xyz_scale_factor) * float(args.y_scale_factor),
                0.006 * float(args.xyz_scale_factor) * float(args.z_scale_factor),
            ),
            rot=(
                0.06 * float(args.rot_scale_factor) * float(args.roll_scale_factor),
                0.06 * float(args.rot_scale_factor) * float(args.pitch_scale_factor),
                0.12 * float(args.rot_scale_factor) * float(args.yaw_scale_factor),
            ),
        )
    episodes = np.asarray(dataset.data["episode_index"], dtype=np.int64)
    unique_eps = np.unique(episodes)

    fold_reports = []
    for ep in unique_eps:
        val_idx = np.where(episodes == ep)[0]
        train_idx = np.where(episodes != ep)[0]
        fold_reports.append(
            _train_fold(
                dataset,
                train_idx,
                val_idx,
                args,
                fold_name=f"heldout_ep_{int(ep)}",
                output_dir=out_dir,
                model_kwargs=model_kwargs,
                init_ckpt=init_ckpt,
            )
        )

    # Final model on all rows for easy downstream evaluation.
    full_train_idx = np.arange(len(dataset), dtype=np.int64)
    full_val_idx = np.arange(len(dataset), dtype=np.int64)
    full_report = _train_fold(
        dataset,
        full_train_idx,
        full_val_idx,
        args,
        fold_name="final_full",
        output_dir=out_dir,
        model_kwargs=model_kwargs,
        init_ckpt=init_ckpt,
    )

    report = {
        "dataset_npz": str(args.dataset_npz),
        "output_dir": str(out_dir),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "proposal_count": int(args.proposal_count),
        "state_dim": int(args.state_dim),
        "hidden_dim": int(args.hidden_dim),
        "init_checkpoint": str(args.init_checkpoint) if args.init_checkpoint else None,
        "train_score_only": bool(args.train_score_only),
        "score_observability_tune": bool(args.score_observability_tune),
        "train_multi_head_score": bool(args.train_multi_head_score),
        "multi_head_selection_mode": str(args.multi_head_selection_mode),
        "input_mode": str(args.input_mode),
        "frontier_cover_weight": float(args.frontier_cover_weight),
        "best_safe_weight": float(args.best_safe_weight),
        "best_geom_weight": float(args.best_geom_weight),
        "yaw_weight": float(args.yaw_weight),
        "yaw_mag_weight": float(args.yaw_mag_weight),
        "action_norm_weight": float(args.action_norm_weight),
        "score_weight": float(args.score_weight),
        "score_pairwise_weight": float(args.score_pairwise_weight),
        "score_utility_alpha": float(args.score_utility_alpha),
        "score_pairwise_margin": float(args.score_pairwise_margin),
        "score_safe_pairwise_weight": float(args.score_safe_pairwise_weight),
        "score_yaw_pairwise_weight": float(args.score_yaw_pairwise_weight),
        "score_ce_weight": float(args.score_ce_weight),
        "score_listwise_weight": float(args.score_listwise_weight),
        "score_target_temperature": float(args.score_target_temperature),
        "score_safe_bonus": float(args.score_safe_bonus),
        "score_pareto_bonus": float(args.score_pareto_bonus),
        "score_safe_risk_budget": float(args.score_safe_risk_budget),
        "score_yaw_bonus": float(args.score_yaw_bonus),
        "multi_best_safe_ce_weight": float(args.multi_best_safe_ce_weight),
        "multi_pareto_bce_weight": float(args.multi_pareto_bce_weight),
        "multi_yaw_bce_weight": float(args.multi_yaw_bce_weight),
        "multi_risk_safe_bce_weight": float(args.multi_risk_safe_bce_weight),
        "multi_geometry_reg_weight": float(args.multi_geometry_reg_weight),
        "multi_utility_rank_weight": float(args.multi_utility_rank_weight),
        "multi_utility_w_safe": float(args.multi_utility_w_safe),
        "multi_utility_w_pareto": float(args.multi_utility_w_pareto),
        "multi_utility_w_yaw": float(args.multi_utility_w_yaw),
        "multi_utility_w_geom": float(args.multi_utility_w_geom),
        "multi_utility_w_risk": float(args.multi_utility_w_risk),
        "score_supervision_mode": str(args.score_supervision_mode),
        "mode_weight": float(args.mode_weight),
        "diversity_weight": float(args.diversity_weight),
        "scale_reg_weight": float(args.scale_reg_weight),
        "yaw_presence_threshold": float(args.yaw_presence_threshold),
        "yaw_match_tol": float(args.yaw_match_tol),
        "yaw_scale_factor": float(args.yaw_scale_factor),
        "rot_scale_factor": float(args.rot_scale_factor),
        "xyz_scale_factor": float(args.xyz_scale_factor),
        "x_scale_factor": float(args.x_scale_factor),
        "y_scale_factor": float(args.y_scale_factor),
        "z_scale_factor": float(args.z_scale_factor),
        "roll_scale_factor": float(args.roll_scale_factor),
        "pitch_scale_factor": float(args.pitch_scale_factor),
        "model_kwargs": model_kwargs,
        "action_scale": asdict(model_kwargs.get("proposal_action_scale", LocalProposalActionScale())),
        "fold_reports": fold_reports,
        "full_report": full_report,
        "mean_fold_val": {
            key: float(np.mean([fold["final_val"].get(key, 0.0) for fold in fold_reports]))
            for key in [
                "selected_best_safe_hit",
                "selected_best_geom_hit",
                "selected_pareto_hit",
                "selected_geometry_gain_mean",
                "selected_risk_delta_mean",
                "selected_yaw_presence_rate",
                "selected_yaw_match_rate",
                "selected_correct_yaw_sign_rate",
                "yaw_loss",
                "score_ce_loss",
                "score_listwise_loss",
                "multi_head_loss",
                "mode_acc",
            ]
        },
    }
    out_json = out_dir / "local_proposal_policy_report.json"
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
