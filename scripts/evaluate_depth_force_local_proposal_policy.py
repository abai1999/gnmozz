#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from prismatic.models.depth_force_local_proposal_policy import DepthForceLocalProposalPolicy
from prismatic.vla.datasets.depth_force_local_proposal_dataset import DepthForceLocalProposalDataset
from local_proposal_utils import evaluate_state_conditioned_proposals, select_best_indices


ACTION_SCALE = torch.tensor([0.008, 0.008, 0.006, 0.06, 0.06, 0.12], dtype=torch.float32)
DEFAULT_YAW_PRESENCE_THRESHOLD = 0.0025
DEFAULT_YAW_MATCH_TOL = 0.0015


def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if hasattr(obj, "__dict__") and obj.__class__.__module__.startswith("prismatic"):
        return {k: _jsonable(v) for k, v in obj.__dict__.items()}
    return obj


def _weighted_l1(a: torch.Tensor, b: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    scale = scale.to(device=a.device, dtype=a.dtype).view(*([1] * (a.ndim - 1)), -1)
    return torch.sum(torch.abs((a - b) / torch.clamp(scale, min=1e-6)), dim=-1)


def _rankdata(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32).reshape(-1)
    order = np.argsort(arr, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float32)
    i = 0
    n = int(arr.size)
    while i < n:
        j = i + 1
        while j < n and arr[order[j]] == arr[order[i]]:
            j += 1
        ranks[order[i:j]] = 0.5 * (i + j - 1) + 1.0
        i = j
    return ranks


def _spearmanr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    y = np.asarray(y, dtype=np.float32).reshape(-1)
    if x.size < 2 or y.size < 2 or x.size != y.size:
        return float("nan")
    rx = _rankdata(x)
    ry = _rankdata(y)
    if np.std(rx) <= 1e-8 or np.std(ry) <= 1e-8:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def _safe_mask(dataset: DepthForceLocalProposalDataset, idx: int) -> np.ndarray:
    pareto = np.asarray(dataset.data["proposal_pareto_mask"][idx], dtype=np.float32) > 0.5
    if np.any(pareto):
        return pareto
    budget = np.asarray(dataset.data["proposal_budget_mask"][idx], dtype=np.float32) > 0.5
    if np.any(budget):
        return budget
    out = np.zeros_like(np.asarray(dataset.data["proposal_pareto_mask"][idx], dtype=np.float32), dtype=bool)
    out[int(dataset.data["proposal_best_safe_index"][idx])] = True
    return out


def _group_summary(rows: np.ndarray, metrics: dict[str, np.ndarray]) -> dict[str, float]:
    idx = np.asarray(rows, dtype=np.int64)
    if idx.size == 0:
        return {"rows": 0}
    out = {"rows": int(idx.size)}
    for key, arr in metrics.items():
        out[key] = float(np.mean(np.asarray(arr)[idx]))
    return out


def _summarize_state_dict_load(
    model: DepthForceLocalProposalPolicy,
    loaded_state_dict: dict[str, torch.Tensor],
) -> dict[str, list[str]]:
    model_keys = set(model.state_dict().keys())
    loaded_keys = set(loaded_state_dict.keys())
    missing_keys = sorted(model_keys - loaded_keys)
    unexpected_keys = sorted(loaded_keys - model_keys)
    return {
        "missing_keys": missing_keys,
        "unexpected_keys": unexpected_keys,
        "loaded_keys": sorted(model_keys & loaded_keys),
        "newly_initialized_keys": list(missing_keys),
    }


def _pairwise_order_changed_rate(base: np.ndarray, pert: np.ndarray) -> float:
    base = np.asarray(base, dtype=np.float32).reshape(-1)
    pert = np.asarray(pert, dtype=np.float32).reshape(-1)
    if base.size <= 1 or pert.size != base.size:
        return 0.0
    base_diff = base[:, None] - base[None, :]
    pert_diff = pert[:, None] - pert[None, :]
    mask = np.triu(np.ones_like(base_diff, dtype=bool), k=1)
    if not np.any(mask):
        return 0.0
    return float(np.mean(np.sign(base_diff[mask]) != np.sign(pert_diff[mask])))


def _make_perturbed_tensor(
    dataset: DepthForceLocalProposalDataset,
    rows: np.ndarray,
    field: str,
    *,
    mode: str,
    base_tensor: torch.Tensor,
    perm_rows: np.ndarray | None = None,
) -> torch.Tensor:
    data = np.asarray(dataset.data[field])
    if mode == "zero":
        return torch.zeros_like(base_tensor)
    if mode == "shuffle":
        if perm_rows is None:
            raise ValueError("perm_rows required for shuffle mode")
        shuffled = data[np.asarray(perm_rows, dtype=np.int64)]
        return torch.as_tensor(shuffled, device=base_tensor.device, dtype=base_tensor.dtype)
    raise ValueError(f"unknown perturbation mode: {mode}")


def _score_sensitivity(
    base_scores: np.ndarray,
    pert_scores: np.ndarray,
) -> dict[str, float]:
    base = np.asarray(base_scores, dtype=np.float32)
    pert = np.asarray(pert_scores, dtype=np.float32)
    if base.shape != pert.shape:
        raise ValueError(f"score shape mismatch: base={base.shape} pert={pert.shape}")
    delta = np.abs(base - pert)
    return {
        "mean_abs_score_delta": float(np.mean(delta)),
        "max_abs_score_delta": float(np.max(delta)),
        "argmax_changed_rate": float(np.mean(np.argmax(base, axis=-1) != np.argmax(pert, axis=-1))),
        "rank_changed_rate": float(
            np.mean([_pairwise_order_changed_rate(b, p) for b, p in zip(base, pert, strict=False)])
        ),
    }


def _select_scores_from_outputs(
    outputs: dict[str, torch.Tensor],
    *,
    selection_mode: str,
    w_safe: float,
    w_pareto: float,
    w_yaw: float,
    w_geom: float,
    w_risk: float,
) -> torch.Tensor:
    mode = str(selection_mode).lower()
    if mode == "scalar":
        return outputs["proposal_scores"]
    multi = outputs["multi_head_scores"]
    if mode == "best_safe":
        return multi[..., 0]
    if mode == "pareto":
        return multi[..., 1]
    if mode == "yaw_match":
        return multi[..., 2]
    if mode == "risk_safe":
        return multi[..., 3]
    if mode == "geometry_gain":
        return multi[..., 4]
    if mode == "weighted_multi":
        return (
            float(w_safe) * multi[..., 0]
            + float(w_pareto) * multi[..., 1]
            + float(w_yaw) * multi[..., 2]
            + float(w_geom) * multi[..., 4]
            - float(w_risk) * multi[..., 3]
        )
    if mode in {"layered_multi", "pareto_then_best_safe", "pareto_then_geometry"}:
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
        masked = torch.where(mask, base, torch.full_like(base, -1e9))
        return masked
    raise ValueError(f"unknown selection_mode={selection_mode!r}")


@torch.no_grad()
def _evaluate_model(
    model: DepthForceLocalProposalPolicy,
    dataset: DepthForceLocalProposalDataset,
    loader: DataLoader,
    *,
    device: torch.device,
    distance_tol: float,
    yaw_presence_threshold: float,
    yaw_match_tol: float,
    sensitivity_audit: bool = False,
    permutation_seed: int = 0,
    selection_mode: str = "scalar",
    multi_utility_w_safe: float = 1.0,
    multi_utility_w_pareto: float = 1.0,
    multi_utility_w_yaw: float = 1.0,
    multi_utility_w_geom: float = 0.5,
    multi_utility_w_risk: float = 0.5,
) -> dict[str, np.ndarray | float]:
    model.eval()
    n = len(dataset)
    best_safe_score_rank = np.full((n,), np.nan, dtype=np.float32)
    best_geom_score_rank = np.full((n,), np.nan, dtype=np.float32)
    pareto_best_score_rank = np.full((n,), np.nan, dtype=np.float32)
    best_safe_score_gap = np.full((n,), np.nan, dtype=np.float32)
    best_geom_score_gap = np.full((n,), np.nan, dtype=np.float32)
    pareto_best_score_gap = np.full((n,), np.nan, dtype=np.float32)
    utility_best_score_rank = np.full((n,), np.nan, dtype=np.float32)
    utility_best_score_gap = np.full((n,), np.nan, dtype=np.float32)
    selected_geom_gain = np.zeros((n,), dtype=np.float32)
    selected_risk_delta = np.zeros((n,), dtype=np.float32)
    selected_yaw_presence_rate = np.zeros((n,), dtype=np.float32)
    selected_yaw_match_rate = np.zeros((n,), dtype=np.float32)
    selected_correct_yaw_sign = np.zeros((n,), dtype=np.float32)
    selected_best_safe_hit = np.zeros((n,), dtype=np.float32)
    selected_best_geom_hit = np.zeros((n,), dtype=np.float32)
    selected_pareto_hit = np.zeros((n,), dtype=np.float32)
    selected_score = np.zeros((n,), dtype=np.float32)
    selected_idx_arr = np.full((n,), -1, dtype=np.int64)
    selected_from_pareto_pool = np.zeros((n,), dtype=np.float32)
    pareto_pool_nonempty = np.zeros((n,), dtype=np.float32)
    fallback_used = np.zeros((n,), dtype=np.float32)
    selected_by_best_safe_head = np.zeros((n,), dtype=np.float32)
    selected_by_geometry_head = np.zeros((n,), dtype=np.float32)
    selected_with_yaw_bonus = np.zeros((n,), dtype=np.float32)
    risk_tiebreak_used = np.zeros((n,), dtype=np.float32)
    set_best_safe_hit = np.zeros((n,), dtype=np.float32)
    set_best_geom_hit = np.zeros((n,), dtype=np.float32)
    set_pareto_hit = np.zeros((n,), dtype=np.float32)
    set_yaw_opportunity_hit = np.zeros((n,), dtype=np.float32)
    set_yaw_match_hit = np.zeros((n,), dtype=np.float32)
    set_correct_yaw_sign_hit = np.zeros((n,), dtype=np.float32)
    set_min_dist_best_safe = np.zeros((n,), dtype=np.float32)
    set_min_dist_best_geom = np.zeros((n,), dtype=np.float32)
    set_best_geometry_gain = np.zeros((n,), dtype=np.float32)
    set_best_risk_delta = np.zeros((n,), dtype=np.float32)
    set_best_safe_geometry_gain = np.zeros((n,), dtype=np.float32)
    set_best_safe_risk_delta = np.zeros((n,), dtype=np.float32)
    score_selected_mode_acc = np.zeros((n,), dtype=np.float32)
    score_selected_risk_nonincrease = np.zeros((n,), dtype=np.float32)
    score_selected_geometry_improve = np.zeros((n,), dtype=np.float32)
    score_selected_yaw_presence_rate = np.zeros((n,), dtype=np.float32)
    score_selected_yaw_match_rate = np.zeros((n,), dtype=np.float32)
    score_selected_correct_yaw_sign = np.zeros((n,), dtype=np.float32)
    oracle_best_geom_gain = np.zeros((n,), dtype=np.float32)
    oracle_best_risk_delta = np.zeros((n,), dtype=np.float32)
    oracle_best_yaw_presence_rate = np.zeros((n,), dtype=np.float32)
    oracle_best_yaw_match_rate = np.zeros((n,), dtype=np.float32)
    oracle_best_correct_yaw_sign = np.zeros((n,), dtype=np.float32)
    score_geom_gain_spearman = np.zeros((n,), dtype=np.float32)
    score_risk_delta_spearman = np.zeros((n,), dtype=np.float32)
    score_utility_spearman = np.zeros((n,), dtype=np.float32)
    sens_metrics: dict[str, list[float]] = {}
    if sensitivity_audit:
        sens_metrics = {
            "zero_depth_mean_abs_score_delta": [],
            "zero_depth_max_abs_score_delta": [],
            "zero_depth_argmax_changed_rate": [],
            "zero_depth_rank_changed_rate": [],
            "shuffle_depth_mean_abs_score_delta": [],
            "shuffle_depth_max_abs_score_delta": [],
            "shuffle_depth_argmax_changed_rate": [],
            "shuffle_depth_rank_changed_rate": [],
            "zero_force_mean_abs_score_delta": [],
            "zero_force_max_abs_score_delta": [],
            "zero_force_argmax_changed_rate": [],
            "zero_force_rank_changed_rate": [],
            "shuffle_force_mean_abs_score_delta": [],
            "shuffle_force_max_abs_score_delta": [],
            "shuffle_force_argmax_changed_rate": [],
            "shuffle_force_rank_changed_rate": [],
            "zero_both_mean_abs_score_delta": [],
            "zero_both_max_abs_score_delta": [],
            "zero_both_argmax_changed_rate": [],
            "zero_both_rank_changed_rate": [],
            "shuffle_both_mean_abs_score_delta": [],
            "shuffle_both_max_abs_score_delta": [],
            "shuffle_both_argmax_changed_rate": [],
            "shuffle_both_rank_changed_rate": [],
        }
        rng = np.random.default_rng(int(permutation_seed))
        depth_perm = rng.permutation(n)
        force_perm = rng.permutation(n)
    fixed_best_safe_geom_gain = np.asarray(dataset.data["proposal_geometry_gain"], dtype=np.float32)[
        np.arange(n), np.asarray(dataset.data["proposal_best_safe_index"], dtype=np.int64)
    ]
    fixed_best_safe_risk_delta = np.asarray(dataset.data["proposal_risk_delta"], dtype=np.float32)[
        np.arange(n), np.asarray(dataset.data["proposal_best_safe_index"], dtype=np.int64)
    ]
    fixed_best_safe_yaw = np.abs(
        np.asarray(dataset.data["proposal_actions_local"], dtype=np.float32)[
            np.arange(n), np.asarray(dataset.data["proposal_best_safe_index"], dtype=np.int64), 5
        ]
    )
    target_delta = np.asarray(dataset.data["proposal_target_delta_local"], dtype=np.float32)
    target_yaw = target_delta[:, 5]
    fixed_best_safe_correct_yaw = np.zeros((n,), dtype=np.float32)
    for i in range(n):
        if abs(float(target_yaw[i])) > float(yaw_presence_threshold):
            yaw = float(dataset.data["proposal_actions_local"][i, dataset.data["proposal_best_safe_index"][i], 5])
            fixed_best_safe_correct_yaw[i] = float((yaw > 0 and target_yaw[i] > 0) or (yaw < 0 and target_yaw[i] < 0))

    for batch in loader:
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        outputs = model(
            front_rgb=batch["front_rgb"],
            wrist_rgb=batch["wrist_rgb"],
            wrist_depth=batch["wrist_depth"],
            force_history=batch["force_history"],
            proprio=batch["proprio"],
            planner_base_action_local=batch["planner_base_action_local"],
            proposal_actions_local=batch["proposal_actions_local"],
            stage_token=batch.get("stage_token"),
            contact_phase=batch.get("contact_phase"),
            depth_proximity=batch.get("depth_proximity"),
            gripper_state=batch.get("gripper_state"),
        )
        proposals = outputs["proposal_actions_local"]
        multi = outputs.get("multi_head_scores", None)
        scores = _select_scores_from_outputs(
            outputs,
            selection_mode=selection_mode,
            w_safe=multi_utility_w_safe,
            w_pareto=multi_utility_w_pareto,
            w_yaw=multi_utility_w_yaw,
            w_geom=multi_utility_w_geom,
            w_risk=multi_utility_w_risk,
        )
        idxs = batch["row_index"].detach().cpu().numpy().astype(np.int64)

        if sensitivity_audit:
            with torch.no_grad():
                zero_depth_out = model(
                    front_rgb=batch["front_rgb"],
                    wrist_rgb=batch["wrist_rgb"],
                    wrist_depth=torch.zeros_like(batch["wrist_depth"]),
                    force_history=batch["force_history"],
                    proprio=batch["proprio"],
                    planner_base_action_local=batch["planner_base_action_local"],
                    proposal_actions_local=batch["proposal_actions_local"],
                    stage_token=batch.get("stage_token"),
                    contact_phase=batch.get("contact_phase"),
                    depth_proximity=batch.get("depth_proximity"),
                    gripper_state=batch.get("gripper_state"),
                )
                zero_depth_out = _select_scores_from_outputs(
                    zero_depth_out,
                    selection_mode=selection_mode,
                    w_safe=multi_utility_w_safe,
                    w_pareto=multi_utility_w_pareto,
                    w_yaw=multi_utility_w_yaw,
                    w_geom=multi_utility_w_geom,
                    w_risk=multi_utility_w_risk,
                ).detach().cpu().numpy()
                shuffle_depth_out = model(
                    front_rgb=batch["front_rgb"],
                    wrist_rgb=batch["wrist_rgb"],
                    wrist_depth=_make_perturbed_tensor(
                        dataset,
                        idxs,
                        "wrist_depth",
                        mode="shuffle",
                        base_tensor=batch["wrist_depth"],
                        perm_rows=depth_perm[idxs],
                    ),
                    force_history=batch["force_history"],
                    proprio=batch["proprio"],
                    planner_base_action_local=batch["planner_base_action_local"],
                    proposal_actions_local=batch["proposal_actions_local"],
                    stage_token=batch.get("stage_token"),
                    contact_phase=batch.get("contact_phase"),
                    depth_proximity=batch.get("depth_proximity"),
                    gripper_state=batch.get("gripper_state"),
                )
                shuffle_depth_out = _select_scores_from_outputs(
                    shuffle_depth_out,
                    selection_mode=selection_mode,
                    w_safe=multi_utility_w_safe,
                    w_pareto=multi_utility_w_pareto,
                    w_yaw=multi_utility_w_yaw,
                    w_geom=multi_utility_w_geom,
                    w_risk=multi_utility_w_risk,
                ).detach().cpu().numpy()
                zero_force_out = model(
                    front_rgb=batch["front_rgb"],
                    wrist_rgb=batch["wrist_rgb"],
                    wrist_depth=batch["wrist_depth"],
                    force_history=torch.zeros_like(batch["force_history"]),
                    proprio=batch["proprio"],
                    planner_base_action_local=batch["planner_base_action_local"],
                    proposal_actions_local=batch["proposal_actions_local"],
                    stage_token=batch.get("stage_token"),
                    contact_phase=batch.get("contact_phase"),
                    depth_proximity=batch.get("depth_proximity"),
                    gripper_state=batch.get("gripper_state"),
                )
                zero_force_out = _select_scores_from_outputs(
                    zero_force_out,
                    selection_mode=selection_mode,
                    w_safe=multi_utility_w_safe,
                    w_pareto=multi_utility_w_pareto,
                    w_yaw=multi_utility_w_yaw,
                    w_geom=multi_utility_w_geom,
                    w_risk=multi_utility_w_risk,
                ).detach().cpu().numpy()
                shuffle_force_out = model(
                    front_rgb=batch["front_rgb"],
                    wrist_rgb=batch["wrist_rgb"],
                    wrist_depth=batch["wrist_depth"],
                    force_history=_make_perturbed_tensor(
                        dataset,
                        idxs,
                        "force_history",
                        mode="shuffle",
                        base_tensor=batch["force_history"],
                        perm_rows=force_perm[idxs],
                    ),
                    proprio=batch["proprio"],
                    planner_base_action_local=batch["planner_base_action_local"],
                    proposal_actions_local=batch["proposal_actions_local"],
                    stage_token=batch.get("stage_token"),
                    contact_phase=batch.get("contact_phase"),
                    depth_proximity=batch.get("depth_proximity"),
                    gripper_state=batch.get("gripper_state"),
                )
                shuffle_force_out = _select_scores_from_outputs(
                    shuffle_force_out,
                    selection_mode=selection_mode,
                    w_safe=multi_utility_w_safe,
                    w_pareto=multi_utility_w_pareto,
                    w_yaw=multi_utility_w_yaw,
                    w_geom=multi_utility_w_geom,
                    w_risk=multi_utility_w_risk,
                ).detach().cpu().numpy()
                zero_both_out = model(
                    front_rgb=batch["front_rgb"],
                    wrist_rgb=batch["wrist_rgb"],
                    wrist_depth=torch.zeros_like(batch["wrist_depth"]),
                    force_history=torch.zeros_like(batch["force_history"]),
                    proprio=batch["proprio"],
                    planner_base_action_local=batch["planner_base_action_local"],
                    proposal_actions_local=batch["proposal_actions_local"],
                    stage_token=batch.get("stage_token"),
                    contact_phase=batch.get("contact_phase"),
                    depth_proximity=batch.get("depth_proximity"),
                    gripper_state=batch.get("gripper_state"),
                )
                zero_both_out = _select_scores_from_outputs(
                    zero_both_out,
                    selection_mode=selection_mode,
                    w_safe=multi_utility_w_safe,
                    w_pareto=multi_utility_w_pareto,
                    w_yaw=multi_utility_w_yaw,
                    w_geom=multi_utility_w_geom,
                    w_risk=multi_utility_w_risk,
                ).detach().cpu().numpy()
                shuffle_both_out = model(
                    front_rgb=batch["front_rgb"],
                    wrist_rgb=batch["wrist_rgb"],
                    wrist_depth=_make_perturbed_tensor(
                        dataset,
                        idxs,
                        "wrist_depth",
                        mode="shuffle",
                        base_tensor=batch["wrist_depth"],
                        perm_rows=depth_perm[idxs],
                    ),
                    force_history=_make_perturbed_tensor(
                        dataset,
                        idxs,
                        "force_history",
                        mode="shuffle",
                        base_tensor=batch["force_history"],
                        perm_rows=force_perm[idxs],
                    ),
                    proprio=batch["proprio"],
                    planner_base_action_local=batch["planner_base_action_local"],
                    proposal_actions_local=batch["proposal_actions_local"],
                    stage_token=batch.get("stage_token"),
                    contact_phase=batch.get("contact_phase"),
                    depth_proximity=batch.get("depth_proximity"),
                    gripper_state=batch.get("gripper_state"),
                )
                shuffle_both_out = _select_scores_from_outputs(
                    shuffle_both_out,
                    selection_mode=selection_mode,
                    w_safe=multi_utility_w_safe,
                    w_pareto=multi_utility_w_pareto,
                    w_yaw=multi_utility_w_yaw,
                    w_geom=multi_utility_w_geom,
                    w_risk=multi_utility_w_risk,
                ).detach().cpu().numpy()
            sens_specs = {
                "zero_depth": zero_depth_out,
                "shuffle_depth": shuffle_depth_out,
                "zero_force": zero_force_out,
                "shuffle_force": shuffle_force_out,
                "zero_both": zero_both_out,
                "shuffle_both": shuffle_both_out,
            }
            base_scores_np = scores.detach().cpu().numpy()
            for key, pert_scores_np in sens_specs.items():
                stats = _score_sensitivity(base_scores_np, pert_scores_np)
                sens_metrics[f"{key}_mean_abs_score_delta"].append(stats["mean_abs_score_delta"])
                sens_metrics[f"{key}_max_abs_score_delta"].append(stats["max_abs_score_delta"])
                sens_metrics[f"{key}_argmax_changed_rate"].append(stats["argmax_changed_rate"])
                sens_metrics[f"{key}_rank_changed_rate"].append(stats["rank_changed_rate"])

        for bi in range(proposals.shape[0]):
            row_idx = int(idxs[bi])
            current_pose = np.asarray(dataset.data.get("privileged_current_pose_7d", dataset.data.get("current_pose_7d"))[row_idx], dtype=np.float32)[None, :]
            target_pose = np.asarray(
                dataset.data.get("privileged_motion_target_pose_7d", dataset.data.get("privileged_basin_center_pose_7d"))[row_idx],
                dtype=np.float32,
            )[None, :]
            cand = proposals[bi : bi + 1].detach().cpu().numpy()
            contact = np.asarray([float(dataset.data.get("contact_label", np.zeros((n,), dtype=np.float32))[row_idx])], dtype=np.float32)
            force_spike = np.asarray([float(dataset.data.get("force_spike_label", np.zeros((n,), dtype=np.float32))[row_idx])], dtype=np.float32)
            jam = np.asarray([float(dataset.data.get("jam_label", np.zeros((n,), dtype=np.float32))[row_idx])], dtype=np.float32)
            stall = np.asarray([float(dataset.data.get("motion_stall_label", np.zeros((n,), dtype=np.float32))[row_idx])], dtype=np.float32)
            near_depth = np.asarray([float(dataset.data.get("near_depth_label", np.zeros((n,), dtype=np.float32))[row_idx])], dtype=np.float32)
            kin_invalid = np.asarray([float(dataset.data.get("kinematic_invalid_label", np.zeros((n,), dtype=np.float32))[row_idx])], dtype=np.float32)
            action_invalid = np.asarray([float(dataset.data.get("action_range_invalid_label", np.zeros((n,), dtype=np.float32))[row_idx])], dtype=np.float32)
            gripper_state = np.asarray([float(batch["gripper_state"][bi].detach().cpu().item())], dtype=np.float32)
            eval_out = evaluate_state_conditioned_proposals(
                current_pose=current_pose,
                target_pose=target_pose,
                candidate_actions=cand,
                contact=contact,
                force_spike=force_spike,
                jam=jam,
                motion_stall=stall,
                near_depth=near_depth,
                kin_invalid=kin_invalid,
                action_range_invalid=action_invalid,
                gripper_state=gripper_state,
            )
            geom = np.asarray(eval_out["candidate_geometry_cost"], dtype=np.float32).reshape(-1)
            risk = np.asarray(eval_out["candidate_risk_cost"], dtype=np.float32).reshape(-1)
            base_geom = float(dataset.data["proposal_geometry_cost"][row_idx, dataset.data["proposal_baseline_index"][row_idx]])
            base_risk = float(dataset.data["proposal_risk_cost"][row_idx, dataset.data["proposal_baseline_index"][row_idx]])
            best_safe_idx = int(dataset.data["proposal_best_safe_index"][row_idx])
            best_geom_idx = int(dataset.data["proposal_geom_top1_index"][row_idx])
            pareto_mask = _safe_mask(dataset, row_idx)
            oracle_actions = np.asarray(dataset.data["proposal_actions_local"][row_idx], dtype=np.float32)
            best_safe_action = torch.from_numpy(dataset.data["proposal_safe_target_action_local"][row_idx]).to(proposals.device).unsqueeze(0)
            best_geom_action = torch.from_numpy(dataset.data["proposal_actions_local"][row_idx, best_geom_idx]).to(proposals.device).unsqueeze(0)
            pareto_actions = torch.from_numpy(dataset.data["proposal_actions_local"][row_idx, pareto_mask]).to(proposals.device) if np.any(pareto_mask) else None

            d_best_safe = _weighted_l1(proposals[bi], best_safe_action, ACTION_SCALE.to(proposals.device))
            d_best_geom = _weighted_l1(proposals[bi], best_geom_action, ACTION_SCALE.to(proposals.device))
            set_min_dist_best_safe[row_idx] = float(torch.min(d_best_safe).item())
            set_min_dist_best_geom[row_idx] = float(torch.min(d_best_geom).item())
            set_best_safe_hit[row_idx] = float(set_min_dist_best_safe[row_idx] <= distance_tol)
            set_best_geom_hit[row_idx] = float(set_min_dist_best_geom[row_idx] <= distance_tol)
            if pareto_actions is not None and pareto_actions.numel() > 0:
                d_pareto = _pairwise = torch.sum(
                    torch.abs(
                        (proposals[bi][:, None, :] - pareto_actions[None, :, :])
                        / torch.clamp(ACTION_SCALE.to(proposals.device).view(1, 1, -1), min=1e-6)
                    ),
                    dim=-1,
                )
                set_pareto_hit[row_idx] = float(torch.min(d_pareto).item() <= distance_tol)
            else:
                set_pareto_hit[row_idx] = 0.0
            tgt_yaw = float(target_yaw[row_idx])
            best_safe_yaw = float(dataset.data["proposal_actions_local"][row_idx, best_safe_idx, 5])
            pred_yaw = proposals[bi][:, 5].detach().cpu().numpy()
            if abs(tgt_yaw) > float(yaw_presence_threshold):
                set_yaw_opportunity_hit[row_idx] = float(np.any(np.abs(pred_yaw) > float(yaw_presence_threshold)))
                set_yaw_match_hit[row_idx] = float(np.any(np.abs(pred_yaw - best_safe_yaw) <= float(yaw_match_tol)))
                set_correct_yaw_sign_hit[row_idx] = float(
                    np.any(((pred_yaw > 0) & (tgt_yaw > 0)) | ((pred_yaw < 0) & (tgt_yaw < 0)))
                )
            else:
                set_yaw_opportunity_hit[row_idx] = 0.0
                set_yaw_match_hit[row_idx] = 0.0
                set_correct_yaw_sign_hit[row_idx] = 0.0
            gen_gain = base_geom - geom
            gen_risk_delta = risk - base_risk
            best_geom_prop_idx = int(np.argmax(gen_gain))
            best_safe_prop_idx = int(np.argmin(d_best_safe.detach().cpu().numpy()))
            set_best_geometry_gain[row_idx] = float(gen_gain[best_geom_prop_idx])
            set_best_risk_delta[row_idx] = float(gen_risk_delta[best_geom_prop_idx])
            set_best_safe_geometry_gain[row_idx] = float(gen_gain[best_safe_prop_idx])
            set_best_safe_risk_delta[row_idx] = float(gen_risk_delta[best_safe_prop_idx])

            selected_idx = int(torch.argmax(scores[bi]).item())
            score_vec = scores[bi].detach().cpu().numpy().astype(np.float32)
            order = np.argsort(-score_vec)
            inv_order = np.empty_like(order)
            inv_order[order] = np.arange(order.size, dtype=np.int64)
            best_safe_gen_idx = int(np.argmin(d_best_safe.detach().cpu().numpy()))
            best_geom_gen_idx = int(np.argmin(d_best_geom.detach().cpu().numpy()))
            utility = (
                np.asarray(base_geom - geom, dtype=np.float32)
                - np.maximum(np.asarray(risk - base_risk, dtype=np.float32), 0.0)
            )
            utility_best_idx = int(np.argmax(utility))
            score_geom_gain_spearman[row_idx] = _spearmanr(score_vec, base_geom - geom)
            score_risk_delta_spearman[row_idx] = _spearmanr(score_vec, -(risk - base_risk))
            score_utility_spearman[row_idx] = _spearmanr(score_vec, utility)
            best_safe_score_rank[row_idx] = float(inv_order[best_safe_gen_idx] + 1)
            best_geom_score_rank[row_idx] = float(inv_order[best_geom_gen_idx] + 1)
            utility_best_score_rank[row_idx] = float(inv_order[utility_best_idx] + 1)
            pareto_best_score_rank[row_idx] = float(inv_order[utility_best_idx] + 1)
            best_safe_score_gap[row_idx] = float(score_vec[selected_idx] - score_vec[best_safe_gen_idx])
            best_geom_score_gap[row_idx] = float(score_vec[selected_idx] - score_vec[best_geom_gen_idx])
            utility_best_score_gap[row_idx] = float(score_vec[selected_idx] - score_vec[utility_best_idx])
            pareto_best_score_gap[row_idx] = float(score_vec[selected_idx] - score_vec[utility_best_idx])
            sel = proposals[bi, selected_idx]
            sel_geom = float(geom[selected_idx])
            sel_risk = float(risk[selected_idx])
            selected_geom_gain[row_idx] = base_geom - sel_geom
            selected_risk_delta[row_idx] = sel_risk - base_risk
            sel_yaw_abs = float(abs(float(sel[5].item())))
            best_safe_yaw_abs = float(abs(best_safe_yaw))
            selected_yaw_presence_rate[row_idx] = float(sel_yaw_abs > float(yaw_presence_threshold))
            selected_yaw_match_rate[row_idx] = float(abs(sel_yaw_abs - best_safe_yaw_abs) <= float(yaw_match_tol))
            tgt_yaw = float(target_yaw[row_idx])
            if abs(tgt_yaw) > float(yaw_presence_threshold):
                selected_correct_yaw_sign[row_idx] = float((float(sel[5].item()) > 0 and tgt_yaw > 0) or (float(sel[5].item()) < 0 and tgt_yaw < 0))
            selected_best_safe_hit[row_idx] = float(
                _weighted_l1(sel.unsqueeze(0), torch.from_numpy(dataset.data["proposal_actions_local"][row_idx, best_safe_idx]).to(sel.device).unsqueeze(0), ACTION_SCALE.to(sel.device))[0].item()
                <= distance_tol
            )
            selected_best_geom_hit[row_idx] = float(
                _weighted_l1(sel.unsqueeze(0), torch.from_numpy(dataset.data["proposal_actions_local"][row_idx, best_geom_idx]).to(sel.device).unsqueeze(0), ACTION_SCALE.to(sel.device))[0].item()
                <= distance_tol
            )
            if np.any(pareto_mask):
                front = torch.from_numpy(dataset.data["proposal_actions_local"][row_idx, pareto_mask]).to(sel.device)
                dists = torch.sum(
                    torch.abs(
                        (sel[None, None, :] - front[:, None, :])
                        / torch.clamp(ACTION_SCALE.to(sel.device).view(1, 1, -1), min=1e-6)
                    ),
                    dim=-1,
                ).squeeze(-1)
                selected_pareto_hit[row_idx] = float(torch.min(dists).item() <= distance_tol)
            else:
                selected_pareto_hit[row_idx] = 0.0
            selected_score[row_idx] = float(scores[bi, selected_idx].item())
            selected_idx_arr[row_idx] = int(selected_idx)
            pareto_pool_nonempty[row_idx] = float(np.any(pareto_mask))
            selected_from_pareto_pool[row_idx] = float(bool(pareto_mask[selected_idx])) if selected_idx < pareto_mask.size else 0.0
            if multi is not None:
                multi_np = multi[bi].detach().cpu().numpy().astype(np.float32)
                best_safe_head_idx = int(np.argmax(multi_np[:, 0]))
                geom_head_idx = int(np.argmax(multi_np[:, 4]))
                selected_by_best_safe_head[row_idx] = float(selected_idx == best_safe_head_idx)
                selected_by_geometry_head[row_idx] = float(selected_idx == geom_head_idx)
                no_yaw_scores = _select_scores_from_outputs(
                    outputs,
                    selection_mode=selection_mode,
                    w_safe=multi_utility_w_safe,
                    w_pareto=multi_utility_w_pareto,
                    w_yaw=0.0,
                    w_geom=multi_utility_w_geom,
                    w_risk=multi_utility_w_risk,
                )[bi].detach().cpu().numpy().astype(np.float32)
                no_risk_scores = _select_scores_from_outputs(
                    outputs,
                    selection_mode=selection_mode,
                    w_safe=multi_utility_w_safe,
                    w_pareto=multi_utility_w_pareto,
                    w_yaw=multi_utility_w_yaw,
                    w_geom=multi_utility_w_geom,
                    w_risk=0.0,
                )[bi].detach().cpu().numpy().astype(np.float32)
                selected_with_yaw_bonus[row_idx] = float(int(np.argmax(no_yaw_scores)) != selected_idx)
                risk_tiebreak_used[row_idx] = float(int(np.argmax(no_risk_scores)) != selected_idx)
                if str(selection_mode).lower() in {"layered_multi", "pareto_then_best_safe", "pareto_then_geometry"}:
                    pareto_prob = torch.sigmoid(multi[bi, :, 1]).detach().cpu().numpy().astype(np.float32)
                    fallback_used[row_idx] = float(np.all(pareto_prob < 0.5))

            # Score-selected mode accuracy and oracle-style measurements.
            score_sel_idx = selected_idx
            score_sel = proposals[bi, score_sel_idx]
            score_sel_geom = float(geom[score_sel_idx])
            score_sel_risk = float(risk[score_sel_idx])
            score_selected_geometry_improve[row_idx] = float(base_geom > score_sel_geom + 1e-6)
            score_selected_risk_nonincrease[row_idx] = float(score_sel_risk <= base_risk + 1e-6)
            score_sel_yaw_abs = float(abs(float(score_sel[5].item())))
            score_selected_yaw_presence_rate[row_idx] = float(score_sel_yaw_abs > float(yaw_presence_threshold))
            score_selected_yaw_match_rate[row_idx] = float(abs(score_sel_yaw_abs - best_safe_yaw_abs) <= float(yaw_match_tol))
            if abs(tgt_yaw) > float(yaw_presence_threshold):
                score_selected_correct_yaw_sign[row_idx] = float(
                    (float(score_sel[5].item()) > 0 and tgt_yaw > 0) or (float(score_sel[5].item()) < 0 and tgt_yaw < 0)
                )
            mode_pred = int(torch.argmax(outputs["mode_logits"][bi]).item())
            mode_true = int(batch["proposal_target_mode_id"][bi].item())
            score_selected_mode_acc[row_idx] = float(mode_true >= 0 and mode_pred == mode_true)

            utility = np.asarray(dataset.data["proposal_geometry_gain"][row_idx], dtype=np.float32) - np.maximum(
                np.asarray(dataset.data["proposal_risk_delta"][row_idx], dtype=np.float32), 0.0
            )
            oracle_best_idx = int(np.argmax(utility))
            oracle_best_geom_gain[row_idx] = float(dataset.data["proposal_geometry_gain"][row_idx, oracle_best_idx])
            oracle_best_risk_delta[row_idx] = float(dataset.data["proposal_risk_delta"][row_idx, oracle_best_idx])
            oracle_best_yaw_abs = float(abs(float(oracle_actions[oracle_best_idx, 5])))
            oracle_best_yaw_presence_rate[row_idx] = float(oracle_best_yaw_abs > float(yaw_presence_threshold))
            oracle_best_yaw_match_rate[row_idx] = float(abs(oracle_best_yaw_abs - best_safe_yaw_abs) <= float(yaw_match_tol))
            if abs(tgt_yaw) > float(yaw_presence_threshold):
                oracle_best_correct_yaw_sign[row_idx] = float(
                    (float(oracle_actions[oracle_best_idx, 5]) > 0 and tgt_yaw > 0)
                    or (float(oracle_actions[oracle_best_idx, 5]) < 0 and tgt_yaw < 0)
                )

    return {
        "selected_geom_gain": selected_geom_gain,
        "selected_risk_delta": selected_risk_delta,
        "selected_yaw_presence_rate": selected_yaw_presence_rate,
        "selected_yaw_match_rate": selected_yaw_match_rate,
        "selected_correct_yaw_sign": selected_correct_yaw_sign,
        "selected_best_safe_hit": selected_best_safe_hit,
        "selected_best_geom_hit": selected_best_geom_hit,
        "selected_pareto_hit": selected_pareto_hit,
        "selected_score": selected_score,
        "selected_idx": selected_idx_arr,
        "selected_from_pareto_pool": selected_from_pareto_pool,
        "pareto_pool_nonempty": pareto_pool_nonempty,
        "fallback_used": fallback_used,
        "selected_by_best_safe_head": selected_by_best_safe_head,
        "selected_by_geometry_head": selected_by_geometry_head,
        "selected_with_yaw_bonus": selected_with_yaw_bonus,
        "risk_tiebreak_used": risk_tiebreak_used,
        "best_safe_score_rank": best_safe_score_rank,
        "best_geom_score_rank": best_geom_score_rank,
        "pareto_best_score_rank": pareto_best_score_rank,
        "best_safe_score_gap": best_safe_score_gap,
        "best_geom_score_gap": best_geom_score_gap,
        "pareto_best_score_gap": pareto_best_score_gap,
        "utility_best_score_rank": utility_best_score_rank,
        "utility_best_score_gap": utility_best_score_gap,
        "set_best_safe_hit": set_best_safe_hit,
        "set_best_geom_hit": set_best_geom_hit,
        "set_pareto_hit": set_pareto_hit,
        "set_yaw_opportunity_hit": set_yaw_opportunity_hit,
        "set_yaw_match_hit": set_yaw_match_hit,
        "set_correct_yaw_sign_hit": set_correct_yaw_sign_hit,
        "set_min_dist_best_safe": set_min_dist_best_safe,
        "set_min_dist_best_geom": set_min_dist_best_geom,
        "set_best_geometry_gain": set_best_geometry_gain,
        "set_best_risk_delta": set_best_risk_delta,
        "set_best_safe_geometry_gain": set_best_safe_geometry_gain,
        "set_best_safe_risk_delta": set_best_safe_risk_delta,
        "score_selected_mode_acc": score_selected_mode_acc,
        "score_selected_risk_nonincrease": score_selected_risk_nonincrease,
        "score_selected_geometry_improve": score_selected_geometry_improve,
        "score_selected_yaw_presence_rate": score_selected_yaw_presence_rate,
        "score_selected_yaw_match_rate": score_selected_yaw_match_rate,
        "score_selected_correct_yaw_sign": score_selected_correct_yaw_sign,
        "oracle_best_geom_gain": oracle_best_geom_gain,
        "oracle_best_risk_delta": oracle_best_risk_delta,
        "oracle_best_yaw_presence_rate": oracle_best_yaw_presence_rate,
        "oracle_best_yaw_match_rate": oracle_best_yaw_match_rate,
        "oracle_best_correct_yaw_sign": oracle_best_correct_yaw_sign,
        "fixed_best_safe_geom_gain": fixed_best_safe_geom_gain,
        "fixed_best_safe_risk_delta": fixed_best_safe_risk_delta,
        "fixed_best_safe_yaw_abs": fixed_best_safe_yaw.astype(np.float32),
        "fixed_best_safe_correct_yaw": fixed_best_safe_correct_yaw,
        "score_geom_gain_spearman": score_geom_gain_spearman,
        "score_risk_delta_spearman": score_risk_delta_spearman,
        "score_utility_spearman": score_utility_spearman,
        "sensitivity": {k: np.asarray(v, dtype=np.float32) for k, v in sens_metrics.items()},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_npz", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--output_json", required=True)
    ap.add_argument("--proposal_cache_npz", default="")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--distance_tol", type=float, default=0.75)
    ap.add_argument("--yaw_presence_threshold", type=float, default=0.0025)
    ap.add_argument("--yaw_match_tol", type=float, default=0.0015)
    ap.add_argument("--sensitivity_audit", action="store_true")
    ap.add_argument("--permutation_seed", type=int, default=0)
    ap.add_argument(
        "--selection_mode",
        default="scalar",
        choices=(
            "scalar",
            "best_safe",
            "pareto",
            "yaw_match",
            "risk_safe",
            "geometry_gain",
            "weighted_multi",
            "layered_multi",
            "pareto_then_best_safe",
            "pareto_then_geometry",
        ),
    )
    ap.add_argument("--multi_utility_w_safe", type=float, default=1.0)
    ap.add_argument("--multi_utility_w_pareto", type=float, default=1.0)
    ap.add_argument("--multi_utility_w_yaw", type=float, default=1.0)
    ap.add_argument("--multi_utility_w_geom", type=float, default=0.5)
    ap.add_argument("--multi_utility_w_risk", type=float, default=0.5)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    dataset = DepthForceLocalProposalDataset(args.dataset_npz, proposal_cache_npz=args.proposal_cache_npz or None)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    inferred_hidden = int(ckpt["model_state_dict"]["proposal_head.0.weight"].shape[0])
    model_kwargs = dict(ckpt.get("model_kwargs", {}))
    model = DepthForceLocalProposalPolicy(
        proposal_count=int(ckpt.get("proposal_count", 8)),
        state_dim=int(ckpt.get("state_dim", 384)),
        hidden_dim=int(ckpt.get("hidden_dim", inferred_hidden)),
        **model_kwargs,
    )
    missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    load_summary = _summarize_state_dict_load(model, ckpt["model_state_dict"])
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
    device = torch.device(args.device)
    model = model.to(device)

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    metrics = _evaluate_model(
        model,
        dataset,
        loader,
        device=device,
        distance_tol=float(args.distance_tol),
        yaw_presence_threshold=float(args.yaw_presence_threshold),
        yaw_match_tol=float(args.yaw_match_tol),
        sensitivity_audit=bool(args.sensitivity_audit),
        permutation_seed=int(args.permutation_seed),
        selection_mode=str(args.selection_mode),
        multi_utility_w_safe=float(args.multi_utility_w_safe),
        multi_utility_w_pareto=float(args.multi_utility_w_pareto),
        multi_utility_w_yaw=float(args.multi_utility_w_yaw),
        multi_utility_w_geom=float(args.multi_utility_w_geom),
        multi_utility_w_risk=float(args.multi_utility_w_risk),
    )
    row_metrics = {k: v for k, v in metrics.items() if k != "sensitivity"}

    n = len(dataset)
    episodes = np.asarray(dataset.data["episode_index"], dtype=np.int64)
    yaw_aug = np.asarray(dataset.data.get("yaw_augmentation_applied", np.zeros((n,), dtype=np.float32)), dtype=np.float32) > 0.5
    yaw_opp = np.asarray(dataset.data.get("yaw_opportunity_label", np.zeros((n,), dtype=np.float32)), dtype=np.float32) > 0.5
    original = ~yaw_aug
    non_yaw = ~yaw_opp
    weak_episodes = np.isin(episodes, np.asarray([1, 8, 10, 19], dtype=np.int64))
    strong_episodes = np.isin(episodes, np.asarray([5, 16, 17, 20], dtype=np.int64))

    report = {
        "dataset_npz": str(args.dataset_npz),
        "checkpoint": str(args.checkpoint),
        "model_kwargs": _jsonable(model_kwargs),
        "load_summary": _jsonable(load_summary),
        "distance_tol": float(args.distance_tol),
        "selection_mode": str(args.selection_mode),
        "multi_utility_weights": {
            "safe": float(args.multi_utility_w_safe),
            "pareto": float(args.multi_utility_w_pareto),
            "yaw": float(args.multi_utility_w_yaw),
            "geom": float(args.multi_utility_w_geom),
            "risk": float(args.multi_utility_w_risk),
        },
        "sensitivity_audit": bool(args.sensitivity_audit),
        "permutation_seed": int(args.permutation_seed),
        "rows": n,
        "all_rows": {
            "selected_geom_gain_mean": float(np.mean(row_metrics["selected_geom_gain"])),
            "selected_risk_delta_mean": float(np.mean(row_metrics["selected_risk_delta"])),
            "selected_yaw_presence_rate": float(np.mean(row_metrics["selected_yaw_presence_rate"])),
            "selected_yaw_match_rate": float(np.mean(row_metrics["selected_yaw_match_rate"])),
            "selected_correct_yaw_sign_rate": float(np.mean(row_metrics["selected_correct_yaw_sign"])),
            "selected_best_safe_hit_rate": float(np.mean(row_metrics["selected_best_safe_hit"])),
            "selected_best_geom_hit_rate": float(np.mean(row_metrics["selected_best_geom_hit"])),
            "selected_pareto_hit_rate": float(np.mean(row_metrics["selected_pareto_hit"])),
            "best_safe_score_rank_mean": float(np.nanmean(row_metrics["best_safe_score_rank"])),
            "best_geom_score_rank_mean": float(np.nanmean(row_metrics["best_geom_score_rank"])),
            "pareto_best_score_rank_mean": float(np.nanmean(row_metrics["pareto_best_score_rank"])),
            "utility_best_score_rank_mean": float(np.nanmean(row_metrics["utility_best_score_rank"])),
            "best_safe_score_gap_mean": float(np.nanmean(row_metrics["best_safe_score_gap"])),
            "best_geom_score_gap_mean": float(np.nanmean(row_metrics["best_geom_score_gap"])),
            "pareto_best_score_gap_mean": float(np.nanmean(row_metrics["pareto_best_score_gap"])),
            "utility_best_score_gap_mean": float(np.nanmean(row_metrics["utility_best_score_gap"])),
            "set_best_safe_recall_at_k": float(np.mean(row_metrics["set_best_safe_hit"])),
            "set_best_geom_recall_at_k": float(np.mean(row_metrics["set_best_geom_hit"])),
            "set_pareto_hit_rate_at_k": float(np.mean(row_metrics["set_pareto_hit"])),
            "set_yaw_opportunity_recall_at_k": float(np.mean(row_metrics["set_yaw_opportunity_hit"])),
            "set_yaw_match_recall_at_k": float(np.mean(row_metrics["set_yaw_match_hit"])),
            "set_correct_yaw_sign_recall_at_k": float(np.mean(row_metrics["set_correct_yaw_sign_hit"])),
            "topk_min_dist_to_best_safe_mean": float(np.mean(row_metrics["set_min_dist_best_safe"])),
            "topk_min_dist_to_best_geom_mean": float(np.mean(row_metrics["set_min_dist_best_geom"])),
            "topk_best_geometry_gain_mean": float(np.mean(row_metrics["set_best_geometry_gain"])),
            "topk_best_risk_delta_mean": float(np.mean(row_metrics["set_best_risk_delta"])),
            "topk_best_safe_geometry_gain_mean": float(np.mean(row_metrics["set_best_safe_geometry_gain"])),
            "topk_best_safe_risk_delta_mean": float(np.mean(row_metrics["set_best_safe_risk_delta"])),
            "yaw_presence_threshold": float(args.yaw_presence_threshold),
            "yaw_match_tol": float(args.yaw_match_tol),
            "score_selected_mode_acc": float(np.mean(row_metrics["score_selected_mode_acc"])),
            "score_selected_risk_nonincrease_rate": float(np.mean(row_metrics["score_selected_risk_nonincrease"])),
            "score_selected_geometry_improve_rate": float(np.mean(row_metrics["score_selected_geometry_improve"])),
            "score_selected_yaw_presence_rate": float(np.mean(row_metrics["score_selected_yaw_presence_rate"])),
            "score_selected_yaw_match_rate": float(np.mean(row_metrics["score_selected_yaw_match_rate"])),
            "score_selected_correct_yaw_sign_rate": float(np.mean(row_metrics["score_selected_correct_yaw_sign"])),
            "oracle_best_geom_gain_mean": float(np.mean(row_metrics["oracle_best_geom_gain"])),
            "oracle_best_risk_delta_mean": float(np.mean(row_metrics["oracle_best_risk_delta"])),
            "oracle_best_yaw_presence_rate": float(np.mean(row_metrics["oracle_best_yaw_presence_rate"])),
            "oracle_best_yaw_match_rate": float(np.mean(row_metrics["oracle_best_yaw_match_rate"])),
            "oracle_best_correct_yaw_sign_rate": float(np.mean(row_metrics["oracle_best_correct_yaw_sign"])),
            "fixed_best_safe_geom_gain_mean": float(np.mean(row_metrics["fixed_best_safe_geom_gain"])),
            "fixed_best_safe_risk_delta_mean": float(np.mean(row_metrics["fixed_best_safe_risk_delta"])),
            "fixed_best_safe_yaw_abs_mean": float(np.mean(row_metrics["fixed_best_safe_yaw_abs"])),
            "fixed_best_safe_yaw_presence_rate": float(np.mean(row_metrics["fixed_best_safe_yaw_abs"] > float(args.yaw_presence_threshold))),
            "fixed_best_safe_correct_yaw_sign_rate": float(np.mean(row_metrics["fixed_best_safe_correct_yaw"])),
        },
        "original_rows": _group_summary(np.where(original)[0], row_metrics),
        "yaw_augmented_rows": _group_summary(np.where(yaw_aug)[0], row_metrics),
        "yaw_opportunity_rows": _group_summary(np.where(yaw_opp)[0], row_metrics),
        "non_yaw_rows": _group_summary(np.where(non_yaw)[0], row_metrics),
        "weak_episodes": _group_summary(np.where(weak_episodes)[0], row_metrics),
        "strong_episodes": _group_summary(np.where(strong_episodes)[0], row_metrics),
    }
    if args.sensitivity_audit:
        sens = metrics.get("sensitivity", {})
        report["sensitivity"] = {
            "zero_depth": {
                "mean_abs_score_delta": float(np.mean(sens.get("zero_depth_mean_abs_score_delta", np.array([0.0], dtype=np.float32)))),
                "max_abs_score_delta": float(np.mean(sens.get("zero_depth_max_abs_score_delta", np.array([0.0], dtype=np.float32)))),
                "argmax_changed_rate": float(np.mean(sens.get("zero_depth_argmax_changed_rate", np.array([0.0], dtype=np.float32)))),
                "rank_changed_rate": float(np.mean(sens.get("zero_depth_rank_changed_rate", np.array([0.0], dtype=np.float32)))),
            },
            "shuffle_depth": {
                "mean_abs_score_delta": float(np.mean(sens.get("shuffle_depth_mean_abs_score_delta", np.array([0.0], dtype=np.float32)))),
                "max_abs_score_delta": float(np.mean(sens.get("shuffle_depth_max_abs_score_delta", np.array([0.0], dtype=np.float32)))),
                "argmax_changed_rate": float(np.mean(sens.get("shuffle_depth_argmax_changed_rate", np.array([0.0], dtype=np.float32)))),
                "rank_changed_rate": float(np.mean(sens.get("shuffle_depth_rank_changed_rate", np.array([0.0], dtype=np.float32)))),
            },
            "zero_force": {
                "mean_abs_score_delta": float(np.mean(sens.get("zero_force_mean_abs_score_delta", np.array([0.0], dtype=np.float32)))),
                "max_abs_score_delta": float(np.mean(sens.get("zero_force_max_abs_score_delta", np.array([0.0], dtype=np.float32)))),
                "argmax_changed_rate": float(np.mean(sens.get("zero_force_argmax_changed_rate", np.array([0.0], dtype=np.float32)))),
                "rank_changed_rate": float(np.mean(sens.get("zero_force_rank_changed_rate", np.array([0.0], dtype=np.float32)))),
            },
            "shuffle_force": {
                "mean_abs_score_delta": float(np.mean(sens.get("shuffle_force_mean_abs_score_delta", np.array([0.0], dtype=np.float32)))),
                "max_abs_score_delta": float(np.mean(sens.get("shuffle_force_max_abs_score_delta", np.array([0.0], dtype=np.float32)))),
                "argmax_changed_rate": float(np.mean(sens.get("shuffle_force_argmax_changed_rate", np.array([0.0], dtype=np.float32)))),
                "rank_changed_rate": float(np.mean(sens.get("shuffle_force_rank_changed_rate", np.array([0.0], dtype=np.float32)))),
            },
            "zero_both": {
                "mean_abs_score_delta": float(np.mean(sens.get("zero_both_mean_abs_score_delta", np.array([0.0], dtype=np.float32)))),
                "max_abs_score_delta": float(np.mean(sens.get("zero_both_max_abs_score_delta", np.array([0.0], dtype=np.float32)))),
                "argmax_changed_rate": float(np.mean(sens.get("zero_both_argmax_changed_rate", np.array([0.0], dtype=np.float32)))),
                "rank_changed_rate": float(np.mean(sens.get("zero_both_rank_changed_rate", np.array([0.0], dtype=np.float32)))),
            },
            "shuffle_both": {
                "mean_abs_score_delta": float(np.mean(sens.get("shuffle_both_mean_abs_score_delta", np.array([0.0], dtype=np.float32)))),
                "max_abs_score_delta": float(np.mean(sens.get("shuffle_both_max_abs_score_delta", np.array([0.0], dtype=np.float32)))),
                "argmax_changed_rate": float(np.mean(sens.get("shuffle_both_argmax_changed_rate", np.array([0.0], dtype=np.float32)))),
                "rank_changed_rate": float(np.mean(sens.get("shuffle_both_rank_changed_rate", np.array([0.0], dtype=np.float32)))),
            },
        }

    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
