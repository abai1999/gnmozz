#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from prismatic.models.depth_force_contact_policy import DepthForceLocalContactPolicy
from prismatic.models.depth_force_future_risk_head import DepthForceCandidateFutureRiskHead
from prismatic.vla.datasets.depth_force_candidate_future_risk_dataset import DepthForceCandidateFutureRiskDataset


def _group_indices(dataset: DepthForceCandidateFutureRiskDataset, indices: np.ndarray) -> dict[str, np.ndarray]:
    yaw_aug = np.asarray(dataset.data.get("yaw_augmentation_applied", np.zeros((dataset.length,), dtype=np.float32)), dtype=np.float32) > 0.5
    yaw_opp = np.asarray(dataset.data.get("yaw_opportunity_label", np.zeros((dataset.length,), dtype=np.float32)), dtype=np.float32) > 0.5
    return {
        "all_rows": np.asarray(indices, dtype=np.int64),
        "original_rows": np.asarray(indices[~yaw_aug[indices]], dtype=np.int64),
        "yaw_augmented_rows": np.asarray(indices[yaw_aug[indices]], dtype=np.int64),
        "yaw_opportunity_rows": np.asarray(indices[yaw_opp[indices]], dtype=np.int64),
        "non_yaw_rows": np.asarray(indices[~yaw_opp[indices]], dtype=np.int64),
    }


def _masked_argmax(scores: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return torch.argmax(scores.masked_fill(~mask, -1e9), dim=1)


def _brier_score(probs: torch.Tensor, labels: torch.Tensor) -> float:
    probs = probs.detach().cpu().to(torch.float32).flatten()
    labels = labels.detach().cpu().to(torch.float32).flatten()
    if probs.numel() == 0:
        return 0.0
    return float(torch.mean((probs - labels) ** 2).item())


def _expected_calibration_error(probs: torch.Tensor, labels: torch.Tensor, bins: int = 10) -> float:
    probs = probs.detach().cpu().to(torch.float32).flatten()
    labels = labels.detach().cpu().to(torch.float32).flatten()
    if probs.numel() == 0:
        return 0.0
    probs_np = probs.numpy()
    labels_np = labels.numpy()
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    total = float(len(probs_np))
    for lo, hi in zip(edges[:-1], edges[1:]):
        if hi < 1.0:
            sel = (probs_np >= lo) & (probs_np < hi)
        else:
            sel = (probs_np >= lo) & (probs_np <= hi)
        if not np.any(sel):
            continue
        conf = float(np.mean(probs_np[sel]))
        acc = float(np.mean(labels_np[sel]))
        ece += (float(np.sum(sel)) / total) * abs(conf - acc)
    return float(ece)


def _select_topk_safe(
    geom_scores: torch.Tensor,
    risk_scores: torch.Tensor,
    mask: torch.Tensor,
    baseline_idx: torch.Tensor,
    topk: int,
    risk_margin: float,
    geo_margin: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    row = torch.arange(geom_scores.shape[0], device=geom_scores.device)
    geom_best = _masked_argmax(geom_scores, mask)
    selected = baseline_idx.clone()
    fallback = torch.zeros_like(selected, dtype=torch.bool)
    for i in range(geom_scores.shape[0]):
        base_geom = float(geom_scores[i, baseline_idx[i]].item())
        base_risk = float(risk_scores[i, baseline_idx[i]].item())
        order = torch.argsort(geom_scores[i].masked_fill(~mask[i], -1e9), descending=True)
        chosen = int(baseline_idx[i].item())
        found = False
        for j in order[: max(1, int(topk))].tolist():
            if not bool(mask[i, j].item()):
                continue
            if float(geom_scores[i, j].item()) >= base_geom + float(geo_margin) and float(risk_scores[i, j].item()) <= base_risk + float(risk_margin):
                chosen = int(j)
                found = True
                break
        selected[i] = chosen
        fallback[i] = not found
    return selected, fallback


def _select_risk_veto(
    geom_scores: torch.Tensor,
    risk_scores: torch.Tensor,
    mask: torch.Tensor,
    baseline_idx: torch.Tensor,
    topk: int,
    risk_margin: float,
    geo_margin: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    geom_top1 = _masked_argmax(geom_scores, mask)
    selected, fallback = _select_topk_safe(geom_scores, risk_scores, mask, baseline_idx, topk, risk_margin, geo_margin)
    row = torch.arange(geom_scores.shape[0], device=geom_scores.device)
    veto = risk_scores[row, geom_top1] > (risk_scores[row, baseline_idx] + float(risk_margin))
    out = torch.where(veto, selected, geom_top1)
    fallback = fallback | (out == baseline_idx)
    return out, fallback


def _select_pareto(
    geom_scores: torch.Tensor,
    risk_scores: torch.Tensor,
    mask: torch.Tensor,
    baseline_idx: torch.Tensor,
    risk_margin: float,
    geo_margin: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    row = torch.arange(geom_scores.shape[0], device=geom_scores.device)
    selected = baseline_idx.clone()
    fallback = torch.zeros_like(selected, dtype=torch.bool)
    for i in range(geom_scores.shape[0]):
        base_geom = float(geom_scores[i, baseline_idx[i]].item())
        base_risk = float(risk_scores[i, baseline_idx[i]].item())
        valid = (
            mask[i]
            & (geom_scores[i] >= base_geom + float(geo_margin))
            & (risk_scores[i] <= base_risk + float(risk_margin))
        )
        if torch.any(valid):
            cand = torch.where(valid)[0]
            # Prefer max geometry improvement, then lowest risk.
            geom_vals = geom_scores[i, cand]
            best_geom = torch.max(geom_vals)
            best_mask = cand[geom_vals >= best_geom - 1e-6]
            if best_mask.numel() > 1:
                best_risk = torch.argmin(risk_scores[i, best_mask])
                chosen = int(best_mask[best_risk].item())
            else:
                chosen = int(best_mask[0].item())
            selected[i] = chosen
        else:
            selected[i] = int(baseline_idx[i].item())
            fallback[i] = True
    return selected, fallback


def _select_soft_topk(
    geom_scores: torch.Tensor,
    risk_scores: torch.Tensor,
    mask: torch.Tensor,
    baseline_idx: torch.Tensor,
    topk: int,
    alpha: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    selected = baseline_idx.clone()
    fallback = torch.zeros_like(selected, dtype=torch.bool)
    for i in range(geom_scores.shape[0]):
        base_geom = float(geom_scores[i, baseline_idx[i]].item())
        base_risk = float(risk_scores[i, baseline_idx[i]].item())
        order = torch.argsort(geom_scores[i].masked_fill(~mask[i], -1e9), descending=True)
        cand = [int(j) for j in order[: max(1, int(topk))].tolist() if bool(mask[i, j].item())]
        if not cand:
            fallback[i] = True
            continue
        geom_gain_vals = torch.tensor(
            [float(geom_scores[i, j].item()) - base_geom for j in cand],
            device=geom_scores.device,
            dtype=torch.float32,
        )
        risk_delta_vals = torch.tensor(
            [max(float(risk_scores[i, j].item()) - base_risk, 0.0) for j in cand],
            device=geom_scores.device,
            dtype=torch.float32,
        )
        geom_gain_std = torch.std(geom_gain_vals, unbiased=False).clamp(min=1e-6)
        risk_delta_std = torch.std(risk_delta_vals, unbiased=False).clamp(min=1e-6)
        geom_gain_norm = (geom_gain_vals - torch.mean(geom_gain_vals)) / geom_gain_std
        risk_delta_norm = (risk_delta_vals - torch.mean(risk_delta_vals)) / risk_delta_std
        best = int(baseline_idx[i].item())
        best_util = -1e18
        for local_j, j in enumerate(cand):
            util = float(geom_gain_norm[local_j].item()) - float(alpha) * float(risk_delta_norm[local_j].item())
            if util > best_util:
                best_util = util
                best = j
        selected[i] = best
        fallback[i] = best == int(baseline_idx[i].item())
    return selected, fallback


def _select_budget_topk(
    geom_scores: torch.Tensor,
    risk_scores: torch.Tensor,
    mask: torch.Tensor,
    baseline_idx: torch.Tensor,
    topk: int,
    risk_budget: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    selected = baseline_idx.clone()
    fallback = torch.zeros_like(selected, dtype=torch.bool)
    for i in range(geom_scores.shape[0]):
        base_risk = float(risk_scores[i, baseline_idx[i]].item())
        order = torch.argsort(geom_scores[i].masked_fill(~mask[i], -1e9), descending=True)
        cand = [int(j) for j in order[: max(1, int(topk))].tolist() if bool(mask[i, j].item())]
        if not cand:
            fallback[i] = True
            continue
        safe = [j for j in cand if (float(risk_scores[i, j].item()) - base_risk) <= float(risk_budget)]
        if safe:
            best = safe[0]
            best_geom = float(geom_scores[i, best].item())
            for j in safe[1:]:
                if float(geom_scores[i, j].item()) > best_geom:
                    best = j
                    best_geom = float(geom_scores[i, j].item())
            selected[i] = int(best)
            fallback[i] = False
            continue
        # No candidate satisfies the budget, so choose the least risky top-k candidate.
        best = cand[0]
        best_risk = float(risk_scores[i, best].item())
        best_geom = float(geom_scores[i, best].item())
        for j in cand[1:]:
            cand_risk = float(risk_scores[i, j].item())
            cand_geom = float(geom_scores[i, j].item())
            if cand_risk < best_risk - 1e-9 or (abs(cand_risk - best_risk) <= 1e-9 and cand_geom > best_geom):
                best = j
                best_risk = cand_risk
                best_geom = cand_geom
        selected[i] = int(best)
        fallback[i] = selected[i] == baseline_idx[i]
    return selected, fallback


def _strategy_metrics(
    name: str,
    selected_idx: torch.Tensor,
    fallback: torch.Tensor,
    geom_scores_pred: torch.Tensor,
    risk_scores_pred: torch.Tensor,
    data: dict[str, torch.Tensor],
    geom_true: torch.Tensor,
    risk_true: torch.Tensor,
    yaw_opp: torch.Tensor,
) -> dict[str, float]:
    row = torch.arange(selected_idx.shape[0], device=selected_idx.device)
    baseline_idx = data["candidate_baseline_index"].to(device=selected_idx.device)
    geom_idx = data["candidate_geom_index"].to(device=selected_idx.device)
    mask = data["candidate_mask"].to(device=selected_idx.device, dtype=torch.bool)
    sel_geom_true = geom_true[row, selected_idx]
    base_geom_true = geom_true[row, baseline_idx]
    sel_risk_true = risk_true[row, selected_idx]
    base_risk_true = risk_true[row, baseline_idx]
    geom_top1 = _masked_argmax(geom_scores_pred, mask)
    geom_top1_geom_true = geom_true[row, geom_top1]
    geom_top1_risk_true = risk_true[row, geom_top1]
    sel_yaw = torch.abs(data["candidate_actions_local"].to(device=selected_idx.device, dtype=torch.float32)[row, selected_idx, 5]) > 0.02
    geom_yaw = torch.abs(data["candidate_actions_local"].to(device=selected_idx.device, dtype=torch.float32)[row, geom_idx, 5]) > 0.02
    base_yaw = torch.abs(data["candidate_actions_local"].to(device=selected_idx.device, dtype=torch.float32)[row, baseline_idx, 5]) > 0.02
    pred_risk_delta = risk_scores_pred[row, selected_idx] - risk_scores_pred[row, baseline_idx]
    pred_geom_delta = geom_scores_pred[row, selected_idx] - geom_scores_pred[row, baseline_idx]
    true_risk_delta = risk_true[row, selected_idx] - risk_true[row, baseline_idx]
    true_geom_delta = geom_true[row, baseline_idx] - geom_true[row, selected_idx]
    pred_risk_prob = torch.sigmoid(pred_risk_delta)
    geom_top1_pred_delta = risk_scores_pred[row, geom_top1] - risk_scores_pred[row, baseline_idx]
    geom_top1_prob = torch.sigmoid(geom_top1_pred_delta)
    out = {
        "rows": float(selected_idx.shape[0]),
        "geometry_improve_rate": float(torch.mean((sel_geom_true < base_geom_true - 1e-6).float()).item()),
        "yaw_success_rate": float(torch.mean(sel_yaw[yaw_opp].float()).item()) if torch.any(yaw_opp) else 0.0,
        "yaw_selected_rate": float(torch.mean(sel_yaw.float()).item()),
        "future_risk_nonincrease_rate": float(torch.mean((sel_risk_true <= base_risk_true + 1e-6).float()).item()),
        "future_risk_reduction_vs_geometry_only_rate": float(torch.mean((sel_risk_true <= geom_top1_risk_true + 1e-6).float()).item()),
        "risk_reduction_vs_geometry_only_mean": float(torch.mean(geom_top1_risk_true - sel_risk_true).item()),
        "risk_reduction_vs_baseline_mean": float(torch.mean(base_risk_true - sel_risk_true).item()),
        "fallback_rate": float(torch.mean(fallback.float()).item()),
        "geometry_regret_delta_mean": float(torch.mean(base_geom_true - sel_geom_true).item()),
        "future_risk_regret_delta_mean": float(torch.mean(base_risk_true - sel_risk_true).item()),
        "geom_top1_geometry_improve_rate": float(torch.mean((geom_top1_geom_true < base_geom_true - 1e-6).float()).item()),
        "geom_top1_risk_nonincrease_rate": float(torch.mean((geom_top1_risk_true <= base_risk_true + 1e-6).float()).item()),
        "geom_top1_yaw_rate": float(torch.mean(geom_yaw.float()).item()),
        "geom_top1_yaw_success_rate": float(torch.mean(geom_yaw[yaw_opp].float()).item()) if torch.any(yaw_opp) else 0.0,
        "baseline_yaw_rate": float(torch.mean(base_yaw.float()).item()),
        "pred_risk_delta_mean": float(torch.mean(pred_risk_delta).item()),
        "pred_geom_delta_mean": float(torch.mean(pred_geom_delta).item()),
        "true_risk_delta_mean": float(torch.mean(true_risk_delta).item()),
        "true_geom_delta_mean": float(torch.mean(true_geom_delta).item()),
        "selected_future_risk_brier": float(_brier_score(pred_risk_prob, true_risk_delta > 0.0)),
        "selected_future_risk_ece": float(_expected_calibration_error(pred_risk_prob, true_risk_delta > 0.0)),
        "geom_top1_future_risk_brier": float(_brier_score(geom_top1_prob, (geom_top1_risk_true - base_risk_true) > 0.0)),
        "geom_top1_future_risk_ece": float(_expected_calibration_error(geom_top1_prob, (geom_top1_risk_true - base_risk_true) > 0.0)),
    }
    geom_top1_geometry_improve_rate = float(out["geom_top1_geometry_improve_rate"])
    geom_top1_yaw_success_rate = float(out["geom_top1_yaw_success_rate"])
    out["geometry_retention_vs_geom_top1"] = float(out["geometry_improve_rate"] / max(geom_top1_geometry_improve_rate, 1e-8))
    out["yaw_retention_vs_geom_top1"] = float(out["yaw_success_rate"] / max(geom_top1_yaw_success_rate, 1e-8))
    out["geometry_vs_geom_top1_delta"] = float(out["geometry_improve_rate"] - geom_top1_geometry_improve_rate)
    out["yaw_vs_geom_top1_delta"] = float(out["yaw_success_rate"] - geom_top1_yaw_success_rate)
    out["future_risk_nonincrease_gain_vs_geom_top1"] = float(out["future_risk_nonincrease_rate"] - float(out["geom_top1_risk_nonincrease_rate"]))
    return out


@torch.no_grad()
def _evaluate_group(
    geom_model: DepthForceLocalContactPolicy,
    risk_model: DepthForceCandidateFutureRiskHead,
    dataset: DepthForceCandidateFutureRiskDataset,
    indices: np.ndarray,
    batch_size: int,
    device: torch.device,
    use_geometry_score_feature: bool,
    topk: int,
    risk_margin: float,
    geo_margin: float,
    soft_alpha: float,
    risk_budget: float,
) -> dict[str, dict[str, float]]:
    if indices.size == 0:
        return {"rows": 0}
    loader = DataLoader(Subset(dataset, indices.tolist()), batch_size=batch_size, shuffle=False, num_workers=0)
    report: dict[str, float] = {}
    weights: dict[str, float] = {}

    def add(name: str, value: float, n: int) -> None:
        if name.endswith(".rows"):
            report[name] = report.get(name, 0.0) + float(value)
            weights[name] = weights.get(name, 0.0) + float(n)
        else:
            report[name] = report.get(name, 0.0) + float(value) * float(n)
            weights[name] = weights.get(name, 0.0) + float(n)

    geom_model.eval()
    risk_model.eval()
    with torch.no_grad():
        for batch in loader:
            front = batch["front_rgb"].to(device=device, dtype=torch.float32)
            wrist = batch["wrist_rgb"].to(device=device, dtype=torch.float32)
            depth = batch["wrist_depth"].to(device=device, dtype=torch.float32)
            force_hist = batch["force_history"].to(device=device, dtype=torch.float32)
            proprio = batch["proprio"].to(device=device, dtype=torch.float32)
            planner = batch["planner_base_action_local"].to(device=device, dtype=torch.float32)
            candidates = batch["candidate_actions_local"].to(device=device, dtype=torch.float32)
            mask = batch["candidate_mask"].to(device=device, dtype=torch.float32) > 0.5
            stage_token = batch["stage_token"].to(device=device)
            contact_phase = batch["contact_phase"].to(device=device)
            depth_prox = batch["depth_proximity"].to(device=device, dtype=torch.float32)
            gripper_state = batch["gripper_state"].to(device=device, dtype=torch.float32)
            baseline_idx = batch["candidate_baseline_index"].to(device=device)
            geom_idx = batch["candidate_geom_index"].to(device=device)
            yaw_opp = batch["yaw_opportunity_label"].to(device=device, dtype=torch.float32) > 0.5
            geom_true = batch["candidate_geometry_cost"].to(device=device, dtype=torch.float32)
            risk_true = batch["candidate_future_risk_score"].to(device=device, dtype=torch.float32)

            geom_out = geom_model(
                front_rgb=front,
                wrist_rgb=wrist,
                wrist_depth=depth,
                force_history=force_hist,
                proprio=proprio,
                planner_base_action_local=planner,
                candidate_actions_local=candidates,
                candidate_mask=mask.float(),
                stage_token=stage_token,
                contact_phase=contact_phase,
                depth_proximity=depth_prox,
                gripper_state=gripper_state,
            )
            geom_scores = geom_out["candidate_geometry_value"].detach()

            risk_scores = risk_model(
                front_rgb=front,
                wrist_rgb=wrist,
                wrist_depth=depth,
                force_history=force_hist,
                proprio=proprio,
                planner_base_action_local=planner,
                candidate_actions_local=candidates,
                candidate_mask=mask.float(),
                geometry_scores=geom_scores if use_geometry_score_feature else None,
                stage_token=stage_token,
                contact_phase=contact_phase,
                depth_proximity=depth_prox,
                gripper_state=gripper_state,
            )["future_total_risk"].detach()

            geom_top1 = _masked_argmax(geom_scores, mask)
            topk_sel, topk_fb = _select_topk_safe(geom_scores, risk_scores, mask, baseline_idx, topk=topk, risk_margin=risk_margin, geo_margin=geo_margin)
            veto_sel, veto_fb = _select_risk_veto(geom_scores, risk_scores, mask, baseline_idx, topk=topk, risk_margin=risk_margin, geo_margin=geo_margin)
            pareto_sel, pareto_fb = _select_pareto(geom_scores, risk_scores, mask, baseline_idx, risk_margin=risk_margin, geo_margin=geo_margin)
            soft_sel, soft_fb = _select_soft_topk(geom_scores, risk_scores, mask, baseline_idx, topk=topk, alpha=soft_alpha)
            budget_sel, budget_fb = _select_budget_topk(geom_scores, risk_scores, mask, baseline_idx, topk=topk, risk_budget=risk_budget)

            strategies = {
                "geom_top1": (geom_top1, torch.zeros_like(geom_top1, dtype=torch.bool)),
                "geom_topk_safe": (topk_sel, topk_fb),
                "risk_veto": (veto_sel, veto_fb),
                "pareto": (pareto_sel, pareto_fb),
                "soft_topk": (soft_sel, soft_fb),
                "budget_topk": (budget_sel, budget_fb),
            }
            for sname, (sel, fb) in strategies.items():
                metrics = _strategy_metrics(sname, sel, fb, geom_scores, risk_scores, batch, geom_true, risk_true, yaw_opp)
                for k, v in metrics.items():
                    add(f"{sname}.{k}", v, int(sel.shape[0]))

    out: dict[str, float] = {}
    for k in report:
        if k.endswith(".rows"):
            out[k] = float(report[k])
        else:
            out[k] = float(report[k] / max(weights.get(k, 1.0), 1.0))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_npz", required=True)
    ap.add_argument("--geometry_checkpoint", required=True)
    ap.add_argument("--future_risk_checkpoint", required=True)
    ap.add_argument("--output_json", default="")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--use_geometry_score_feature", action="store_true", default=False)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--risk_margin", type=float, default=0.0)
    ap.add_argument("--geo_margin", type=float, default=0.0)
    ap.add_argument("--soft_alpha", type=float, default=0.3)
    ap.add_argument("--risk_budget", type=float, default=0.05)
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    dataset = DepthForceCandidateFutureRiskDataset(args.dataset_npz)
    geom_ckpt = torch.load(args.geometry_checkpoint, map_location="cpu")
    geom_model = DepthForceLocalContactPolicy().to(device)
    geom_model.load_state_dict(geom_ckpt["model_state_dict"], strict=True)
    geom_model.eval()

    risk_ckpt = torch.load(args.future_risk_checkpoint, map_location="cpu")
    risk_model = DepthForceCandidateFutureRiskHead().to(device)
    risk_model.load_state_dict(risk_ckpt["model_state_dict"], strict=True)
    risk_model.eval()

    eps = np.asarray(dataset.data.get("episode_index", np.zeros((dataset.length,), dtype=np.int64)), dtype=np.int64)
    base_indices = np.arange(dataset.length, dtype=np.int64)
    groups = _group_indices(dataset, base_indices)

    report: dict[str, object] = {
        "dataset_npz": args.dataset_npz,
        "geometry_checkpoint": args.geometry_checkpoint,
        "future_risk_checkpoint": args.future_risk_checkpoint,
        "use_geometry_score_feature": bool(args.use_geometry_score_feature),
        "topk": int(args.topk),
        "risk_margin": float(args.risk_margin),
        "geo_margin": float(args.geo_margin),
        "soft_alpha": float(args.soft_alpha),
        "risk_budget": float(args.risk_budget),
        "groups": {},
        "episodes": {},
        "strategy_names": ["geom_top1", "geom_topk_safe", "risk_veto", "pareto", "soft_topk", "budget_topk"],
    }
    for gname, idx in groups.items():
        report["groups"][gname] = _evaluate_group(
            geom_model=geom_model,
            risk_model=risk_model,
            dataset=dataset,
            indices=np.asarray(idx, dtype=np.int64),
            batch_size=args.batch_size,
            device=device,
            use_geometry_score_feature=bool(args.use_geometry_score_feature),
            topk=int(args.topk),
            risk_margin=float(args.risk_margin),
            geo_margin=float(args.geo_margin),
            soft_alpha=float(args.soft_alpha),
            risk_budget=float(args.risk_budget),
        )

    for ep in sorted(int(x) for x in np.unique(eps)):
        idx = np.where(eps == ep)[0]
        report["episodes"][str(ep)] = _evaluate_group(
            geom_model=geom_model,
            risk_model=risk_model,
            dataset=dataset,
            indices=np.asarray(idx, dtype=np.int64),
            batch_size=args.batch_size,
            device=device,
            use_geometry_score_feature=bool(args.use_geometry_score_feature),
            topk=int(args.topk),
            risk_margin=float(args.risk_margin),
            geo_margin=float(args.geo_margin),
            soft_alpha=float(args.soft_alpha),
            risk_budget=float(args.risk_budget),
        )

    out_json = Path(args.output_json) if args.output_json else Path(args.future_risk_checkpoint).with_name("geometry_future_risk_selector_report.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
