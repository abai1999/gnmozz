#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from prismatic.models.depth_force_contact_policy import DepthForceLocalContactPolicy
from scripts.train_depth_force_mode_first_geometry_risk_policy import PrivilegedGeometryCandidateDataset


def _pick_cost(batch: dict[str, torch.Tensor], raw_key: str, norm_key: str, use_normalized: bool) -> torch.Tensor:
    if use_normalized and norm_key in batch:
        return batch[norm_key]
    return batch[raw_key]


def _load_model(checkpoint_path: str, device: torch.device) -> DepthForceLocalContactPolicy:
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    model = DepthForceLocalContactPolicy().to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()
    return model


def _group_indices(dataset: PrivilegedGeometryCandidateDataset, indices: np.ndarray) -> dict[str, np.ndarray]:
    yaw_aug = np.asarray(dataset.data.get("yaw_augmentation_applied", np.zeros((dataset.length,), dtype=np.float32)), dtype=np.float32) > 0.5
    yaw_opp = np.asarray(dataset.data.get("yaw_opportunity_label", np.zeros((dataset.length,), dtype=np.float32)), dtype=np.float32) > 0.5
    return {
        "all_rows": np.asarray(indices, dtype=np.int64),
        "original_rows": np.asarray(indices[~yaw_aug[indices]], dtype=np.int64),
        "yaw_augmented_rows": np.asarray(indices[yaw_aug[indices]], dtype=np.int64),
        "yaw_opportunity_rows": np.asarray(indices[yaw_opp[indices]], dtype=np.int64),
        "non_yaw_rows": np.asarray(indices[~yaw_opp[indices]], dtype=np.int64),
    }


def _select_with_risk_veto(
    geom_scores: torch.Tensor,
    risk_selector: torch.Tensor,
    mask: torch.Tensor,
    baseline_idx: torch.Tensor,
    margin: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Geometry-first selection, vetoed by risk.

    Returns:
        selected_idx, geom_best_idx, veto_triggered
    """
    bsz, cand = geom_scores.shape
    safe_geom = geom_scores.masked_fill(~mask, -1e9)
    geom_best_idx = torch.argmax(safe_geom, dim=1)
    selected_idx = baseline_idx.clone()
    veto_triggered = torch.zeros(bsz, dtype=torch.bool, device=geom_scores.device)
    for i in range(bsz):
        base_risk = risk_selector[i, baseline_idx[i]]
        order = torch.argsort(safe_geom[i], descending=True)
        chosen = int(baseline_idx[i].item())
        for j in order.tolist():
            if not bool(mask[i, j].item()):
                continue
            if float(risk_selector[i, j].item()) <= float(base_risk.item()) + float(margin):
                chosen = int(j)
                break
        selected_idx[i] = chosen
        veto_triggered[i] = chosen != int(geom_best_idx[i].item())
    return selected_idx, geom_best_idx, veto_triggered


def _evaluate_selection(
    batch: dict[str, torch.Tensor],
    selected_idx: torch.Tensor,
    geom_best_idx: torch.Tensor,
    geom_scores: torch.Tensor,
    risk_selector: torch.Tensor,
    use_normalized_costs: bool,
) -> dict[str, float]:
    row = torch.arange(selected_idx.shape[0], device=selected_idx.device)
    mask = batch["candidate_mask"].to(device=selected_idx.device, dtype=torch.bool)
    total_cost = _pick_cost(batch, "candidate_total_cost", "candidate_total_cost_norm", use_normalized_costs).to(selected_idx.device)
    geom_cost = _pick_cost(batch, "candidate_geometry_cost", "candidate_geometry_cost_norm", use_normalized_costs).to(selected_idx.device)
    risk_cost = _pick_cost(batch, "candidate_risk_cost", "candidate_risk_cost_norm", use_normalized_costs).to(selected_idx.device)
    baseline_idx = batch["candidate_baseline_index"].to(device=selected_idx.device)
    yaw_opp = batch["yaw_opportunity_label"].to(device=selected_idx.device, dtype=torch.float32) > 0.5
    yaw_sel = torch.abs(
        batch["candidate_actions_local"].to(device=selected_idx.device, dtype=torch.float32)[row, selected_idx, 5]
    ) > 0.02
    geom_sel = geom_cost[row, selected_idx]
    base_geom = geom_cost[row, baseline_idx]
    total_sel = total_cost[row, selected_idx]
    base_total = total_cost[row, baseline_idx]
    risk_sel = risk_cost[row, selected_idx]
    base_risk = risk_cost[row, baseline_idx]

    geom_best_geom = geom_cost[row, geom_best_idx]
    geom_best_total = total_cost[row, geom_best_idx]
    geom_best_risk = risk_cost[row, geom_best_idx]
    geom_worst_idx = torch.argmin(geom_scores.masked_fill(~mask, 1e9), dim=1)
    total_worst_idx = torch.argmin(total_cost.masked_fill(~mask, 1e9), dim=1)

    selected_is_geom_best = selected_idx == geom_best_idx
    geom_best_risk_violation = risk_selector[row, geom_best_idx] > (risk_selector[row, baseline_idx] + 1e-6)
    selected_risk_viol = risk_selector[row, selected_idx] > (risk_selector[row, baseline_idx] + 1e-6)

    out = {
        "rows": float(selected_idx.shape[0]),
        "selected_geometry_improves_rate": float(torch.mean((geom_sel < base_geom - 1e-6).float()).item()),
        "selected_total_improves_rate": float(torch.mean((total_sel < base_total - 1e-6).float()).item()),
        "risk_non_increase_rate": float(torch.mean((risk_sel <= base_risk + 1e-6).float()).item()),
        "geometry_regret_delta_mean": float(torch.mean(base_geom - geom_sel).item()),
        "total_regret_delta_mean": float(torch.mean(base_total - total_sel).item()),
        "risk_delta_mean": float(torch.mean(base_risk - risk_sel).item()),
        "yaw_opportunity_selected_rate": float(torch.mean(yaw_sel[yaw_opp].float()).item()) if torch.any(yaw_opp) else 0.0,
        "geom_best_selected_rate": float(torch.mean(selected_is_geom_best.float()).item()),
        "geom_best_risk_violation_rate": float(torch.mean(geom_best_risk_violation.float()).item()),
        "selected_risk_violation_rate": float(torch.mean(selected_risk_viol.float()).item()),
        "selected_yaw_rate": float(torch.mean(yaw_sel.float()).item()),
        "geom_best_yaw_rate": float(
            torch.mean(
                (
                    torch.abs(
                        batch["candidate_actions_local"].to(device=selected_idx.device, dtype=torch.float32)[row, geom_best_idx, 5]
                    )
                    > 0.02
                ).float()
            ).item()
        ),
        "geometry_score_best_minus_baseline": float(
            torch.mean(geom_scores[row, geom_best_idx] - geom_scores[row, baseline_idx]).item()
        ),
        "geometry_score_best_minus_worst": float(
            torch.mean(geom_scores[row, geom_best_idx] - geom_scores[row, geom_worst_idx]).item()
        ),
        "total_score_best_minus_baseline": float(
            torch.mean(
                batch["candidate_total_cost"].to(device=selected_idx.device)[row, geom_best_idx]
                - batch["candidate_total_cost"].to(device=selected_idx.device)[row, baseline_idx]
            ).item()
        ),
        "total_score_best_minus_worst": float(
            torch.mean(batch["candidate_total_cost"].to(device=selected_idx.device)[row, geom_best_idx] - batch["candidate_total_cost"].to(device=selected_idx.device)[row, total_worst_idx]).item()
        ),
    }
    return out


@torch.no_grad()
def _collect_cache(
    model: DepthForceLocalContactPolicy,
    dataset: PrivilegedGeometryCandidateDataset,
    indices: np.ndarray,
    batch_size: int,
    device: torch.device,
    use_normalized_costs: bool,
) -> dict[str, torch.Tensor]:
    if indices.size == 0:
        return {"rows": torch.zeros(0, dtype=torch.long)}
    loader = DataLoader(Subset(dataset, indices.tolist()), batch_size=batch_size, shuffle=False, num_workers=0)
    chunks: dict[str, list[torch.Tensor]] = {
        "geom_scores": [],
        "risk_scores_pred": [],
        "total_cost": [],
        "geom_cost": [],
        "risk_cost": [],
        "baseline_idx": [],
        "episode_index": [],
        "mask": [],
        "candidate_actions_local": [],
        "yaw_opp": [],
        "yaw_aug": [],
    }
    for batch in loader:
        front = batch["front_rgb"].to(device=device, dtype=torch.float32)
        wrist = batch["wrist_rgb"].to(device=device, dtype=torch.float32)
        depth = batch["wrist_depth"].to(device=device, dtype=torch.float32)
        force_hist = batch["force_history"].to(device=device, dtype=torch.float32)
        proprio = batch["proprio"].to(device=device, dtype=torch.float32)
        planner = batch["planner_base_action_local"].to(device=device, dtype=torch.float32)
        candidates = batch["candidate_actions_local"].to(device=device, dtype=torch.float32)
        mask = batch["candidate_mask"].to(device=device, dtype=torch.bool)
        stage_token = batch["stage_token"].to(device=device)
        contact_phase = batch["contact_phase"].to(device=device)
        depth_prox = batch["depth_proximity"].to(device=device, dtype=torch.float32)
        gripper_state = batch["gripper_state"].to(device=device, dtype=torch.float32)
        out = model(
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
        chunks["geom_scores"].append(out["candidate_geometry_value"].detach().cpu())
        chunks["risk_scores_pred"].append(out["candidate_risk_value"].detach().cpu())
        chunks["total_cost"].append(_pick_cost(batch, "candidate_total_cost", "candidate_total_cost_norm", use_normalized_costs).detach().cpu())
        chunks["geom_cost"].append(_pick_cost(batch, "candidate_geometry_cost", "candidate_geometry_cost_norm", use_normalized_costs).detach().cpu())
        chunks["risk_cost"].append(_pick_cost(batch, "candidate_risk_cost", "candidate_risk_cost_norm", use_normalized_costs).detach().cpu())
        chunks["baseline_idx"].append(batch["candidate_baseline_index"].detach().cpu())
        chunks["episode_index"].append(batch["episode_index"].detach().cpu())
        chunks["mask"].append(mask.detach().cpu())
        chunks["candidate_actions_local"].append(batch["candidate_actions_local"].detach().cpu())
        chunks["yaw_opp"].append(batch["yaw_opportunity_label"].detach().cpu() > 0.5)
        chunks["yaw_aug"].append(batch["yaw_augmentation_applied"].detach().cpu() > 0.5)
    return {k: torch.cat(v, dim=0) for k, v in chunks.items()}


def _evaluate_from_cache(
    cache: dict[str, torch.Tensor],
    risk_source: str,
    margin: float,
) -> dict[str, object]:
    if cache["geom_scores"].numel() == 0:
        return {"rows": 0}
    geom_scores = cache["geom_scores"]
    risk_selector = cache["risk_scores_pred"] if risk_source == "predicted" else cache["risk_cost"]
    mask = cache["mask"].bool()
    baseline_idx = cache["baseline_idx"].long()
    selected_geom_idx, geom_best_idx, _ = _select_with_risk_veto(geom_scores, risk_selector, mask, baseline_idx, margin=1e9)
    veto_selected_idx, _, veto_triggered = _select_with_risk_veto(geom_scores, risk_selector, mask, baseline_idx, margin=margin)

    def _metrics(selected_idx: torch.Tensor) -> dict[str, float]:
        row = torch.arange(selected_idx.shape[0])
        total_cost = cache["total_cost"]
        geom_cost = cache["geom_cost"]
        risk_cost = cache["risk_cost"]
        yaw_opp = cache["yaw_opp"]
        yaw_sel = torch.abs(cache["candidate_actions_local"][row, selected_idx, 5]) > 0.02
        geom_sel = geom_cost[row, selected_idx]
        base_geom = geom_cost[row, baseline_idx]
        total_sel = total_cost[row, selected_idx]
        base_total = total_cost[row, baseline_idx]
        risk_sel = risk_cost[row, selected_idx]
        base_risk = risk_cost[row, baseline_idx]
        geom_best_geom = geom_cost[row, geom_best_idx]
        geom_best_total = total_cost[row, geom_best_idx]
        geom_best_risk = risk_cost[row, geom_best_idx]
        geom_worst_idx = torch.argmin(geom_scores.masked_fill(~mask, 1e9), dim=1)
        total_worst_idx = torch.argmin(total_cost.masked_fill(~mask, 1e9), dim=1)
        selected_is_geom_best = selected_idx == geom_best_idx
        geom_best_risk_violation = risk_selector[row, geom_best_idx] > (risk_selector[row, baseline_idx] + 1e-6)
        selected_risk_viol = risk_selector[row, selected_idx] > (risk_selector[row, baseline_idx] + 1e-6)
        return {
            "rows": float(selected_idx.shape[0]),
            "selected_geometry_improves_rate": float(torch.mean((geom_sel < base_geom - 1e-6).float()).item()),
            "selected_total_improves_rate": float(torch.mean((total_sel < base_total - 1e-6).float()).item()),
            "risk_non_increase_rate": float(torch.mean((risk_sel <= base_risk + 1e-6).float()).item()),
            "geometry_regret_delta_mean": float(torch.mean(base_geom - geom_sel).item()),
            "total_regret_delta_mean": float(torch.mean(base_total - total_sel).item()),
            "risk_delta_mean": float(torch.mean(base_risk - risk_sel).item()),
            "yaw_opportunity_selected_rate": float(torch.mean(yaw_sel[yaw_opp].float()).item()) if torch.any(yaw_opp) else 0.0,
            "geom_best_selected_rate": float(torch.mean(selected_is_geom_best.float()).item()),
            "geom_best_risk_violation_rate": float(torch.mean(geom_best_risk_violation.float()).item()),
            "selected_risk_violation_rate": float(torch.mean(selected_risk_viol.float()).item()),
            "selected_yaw_rate": float(torch.mean(yaw_sel.float()).item()),
            "geom_best_yaw_rate": float(
                torch.mean((torch.abs(cache["candidate_actions_local"][row, geom_best_idx, 5]) > 0.02).float()).item()
            ),
            "geometry_score_best_minus_baseline": float(torch.mean(geom_scores[row, geom_best_idx] - geom_scores[row, baseline_idx]).item()),
            "geometry_score_best_minus_worst": float(torch.mean(geom_scores[row, geom_best_idx] - geom_scores[row, geom_worst_idx]).item()),
            "total_score_best_minus_baseline": float(torch.mean(total_cost[row, geom_best_idx] - total_cost[row, baseline_idx]).item()),
            "total_score_best_minus_worst": float(torch.mean(total_cost[row, geom_best_idx] - total_cost[row, total_worst_idx]).item()),
        }

    geom_metrics = _metrics(selected_geom_idx)
    veto_metrics = _metrics(veto_selected_idx)
    veto_metrics["veto_trigger_rate"] = float(torch.mean(veto_triggered.float()).item())
    veto_metrics["fallback_to_baseline_rate"] = float(torch.mean((veto_selected_idx == baseline_idx).float()).item())
    veto_metrics["veto_recovered_rate"] = float(torch.mean(((veto_selected_idx != baseline_idx) & (veto_selected_idx != geom_best_idx)).float()).item())
    return {
        "rows": float(cache["geom_scores"].shape[0]),
        "risk_source": risk_source,
        "margin": float(margin),
        "geom": geom_metrics,
        "veto": veto_metrics,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_npz", required=True)
    ap.add_argument("--checkpoint_path", required=True)
    ap.add_argument("--output_json", default="")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--use_normalized_costs", action="store_true", default=False)
    ap.add_argument("--risk_sources", nargs="*", default=["oracle", "predicted"], choices=["oracle", "predicted"])
    ap.add_argument("--veto_margins", nargs="*", type=float, default=[0.0, 0.05, 0.1])
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    dataset = PrivilegedGeometryCandidateDataset(args.dataset_npz)
    model = _load_model(args.checkpoint_path, device)

    eps = np.asarray(dataset.data.get("episode_index", np.zeros((dataset.length,), dtype=np.int64)), dtype=np.int64)
    yaw_aug = np.asarray(dataset.data.get("yaw_augmentation_applied", np.zeros((dataset.length,), dtype=np.float32)), dtype=np.float32) > 0.5
    yaw_opp = np.asarray(dataset.data.get("yaw_opportunity_label", np.zeros((dataset.length,), dtype=np.float32)), dtype=np.float32) > 0.5
    base_indices = np.arange(dataset.length, dtype=np.int64)
    groups = {
        "all_rows": base_indices,
        "original_rows": base_indices[~yaw_aug],
        "yaw_augmented_rows": base_indices[yaw_aug],
        "yaw_opportunity_rows": base_indices[yaw_opp],
        "non_yaw_rows": base_indices[~yaw_opp],
    }

    report: dict[str, object] = {
        "dataset_npz": args.dataset_npz,
        "checkpoint_path": args.checkpoint_path,
        "use_normalized_costs": bool(args.use_normalized_costs),
        "risk_sources": list(args.risk_sources),
        "veto_margins": list(args.veto_margins),
        "groups": {},
        "episodes": {},
    }
    for gname, idx in groups.items():
        cache = _collect_cache(
            model=model,
            dataset=dataset,
            indices=np.asarray(idx, dtype=np.int64),
            batch_size=args.batch_size,
            device=device,
            use_normalized_costs=args.use_normalized_costs,
        )
        report["groups"][gname] = {}
        for risk_source in args.risk_sources:
            for margin in args.veto_margins:
                key = f"{risk_source}_margin_{margin:g}"
                report["groups"][gname][key] = _evaluate_from_cache(cache, risk_source=risk_source, margin=float(margin))

    chosen_source = args.risk_sources[0]
    chosen_margin = float(args.veto_margins[0])
    for ep in sorted(int(x) for x in np.unique(eps)):
        idx = np.where(eps == ep)[0]
        cache = _collect_cache(
            model=model,
            dataset=dataset,
            indices=np.asarray(idx, dtype=np.int64),
            batch_size=args.batch_size,
            device=device,
            use_normalized_costs=args.use_normalized_costs,
        )
        report["episodes"][str(ep)] = _evaluate_from_cache(cache, risk_source=chosen_source, margin=chosen_margin)

    out_json = Path(args.output_json) if args.output_json else Path(args.checkpoint_path).with_name("geometry_risk_veto_report.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
