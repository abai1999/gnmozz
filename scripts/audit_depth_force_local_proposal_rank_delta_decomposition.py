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


def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj


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


def _rank_desc(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float32)
    order = np.argsort(-scores, axis=-1, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.int64)
    row = np.arange(scores.shape[0])[:, None]
    ranks[row, order] = np.arange(1, scores.shape[1] + 1, dtype=np.int64)
    return ranks


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


def _candidatewise_delta_summary(base_scores: np.ndarray, pert_scores: np.ndarray) -> dict[str, object]:
    base = np.asarray(base_scores, dtype=np.float32)
    pert = np.asarray(pert_scores, dtype=np.float32)
    if base.shape != pert.shape:
        raise ValueError(f"score shape mismatch: base={base.shape} pert={pert.shape}")
    delta = pert - base
    row_std = np.std(delta, axis=1)
    row_range = np.max(delta, axis=1) - np.min(delta, axis=1)

    base_rank = _rank_desc(base)
    pert_rank = _rank_desc(pert)
    rank_delta = pert_rank - base_rank

    return {
        "score_delta_per_candidate_mean": np.mean(delta, axis=0).astype(np.float32),
        "score_delta_per_candidate_std": np.std(delta, axis=0).astype(np.float32),
        "candidatewise_rank_delta_mean": np.mean(rank_delta, axis=0).astype(np.float32),
        "candidatewise_rank_delta_std": np.std(rank_delta, axis=0).astype(np.float32),
        "within_row_delta_std_mean": float(np.mean(row_std)),
        "within_row_delta_std_std": float(np.std(row_std)),
        "within_row_delta_range_mean": float(np.mean(row_range)),
        "within_row_delta_range_std": float(np.std(row_range)),
        "within_row_delta_mean_abs_mean": float(np.mean(np.abs(delta))),
        "within_row_delta_std_to_mean_abs_ratio": float(np.mean(row_std) / max(float(np.mean(np.abs(delta))), 1e-8)),
        "within_row_delta_range_to_mean_abs_ratio": float(np.mean(row_range) / max(float(np.mean(np.abs(delta))), 1e-8)),
        "argmax_changed_rate": float(np.mean(np.argmax(base, axis=-1) != np.argmax(pert, axis=-1))),
        "rank_changed_rate": float(np.mean([_pairwise_order_changed_rate(b, p) for b, p in zip(base, pert, strict=False)])),
    }


def _select_scores(
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
        return torch.where(mask, base, torch.full_like(base, -1e9))
    raise ValueError(f"unknown selection_mode={selection_mode!r}")


@torch.no_grad()
def _load_model(
    checkpoint: str,
    *,
    device: torch.device,
) -> tuple[DepthForceLocalProposalPolicy, dict[str, object]]:
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
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
    return model.to(device), {
        "checkpoint": checkpoint,
        "model_kwargs": model_kwargs,
        "load_summary": load_summary,
    }


@torch.no_grad()
def _evaluate(
    model: DepthForceLocalProposalPolicy,
    dataset: DepthForceLocalProposalDataset,
    loader: DataLoader,
    *,
    device: torch.device,
    sensitivity_audit: bool,
    permutation_seed: int,
    selection_mode: str,
    multi_utility_w_safe: float,
    multi_utility_w_pareto: float,
    multi_utility_w_yaw: float,
    multi_utility_w_geom: float,
    multi_utility_w_risk: float,
) -> dict[str, object]:
    model.eval()
    n = len(dataset)
    base_scores_all: list[np.ndarray] = []
    row_index_all: list[np.ndarray] = []
    outputs_cache: dict[str, list[np.ndarray]] = {
        "zero_depth": [],
        "shuffle_depth": [],
        "zero_force": [],
        "shuffle_force": [],
        "zero_both": [],
        "shuffle_both": [],
    }
    if sensitivity_audit:
        rng = np.random.default_rng(int(permutation_seed))
        depth_perm = rng.permutation(n)
        force_perm = rng.permutation(n)
    else:
        depth_perm = np.arange(n, dtype=np.int64)
        force_perm = np.arange(n, dtype=np.int64)

    for batch in loader:
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        base_out = model(
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
        base = _select_scores(
            base_out,
            selection_mode=selection_mode,
            w_safe=multi_utility_w_safe,
            w_pareto=multi_utility_w_pareto,
            w_yaw=multi_utility_w_yaw,
            w_geom=multi_utility_w_geom,
            w_risk=multi_utility_w_risk,
        ).detach().cpu().numpy()
        base_scores_all.append(base)
        row_index_all.append(batch["row_index"].detach().cpu().numpy())

        if sensitivity_audit:
            idxs = batch["row_index"].detach().cpu().numpy().astype(np.int64)
            perturb_specs = {
                "zero_depth": dict(
                    wrist_depth=torch.zeros_like(batch["wrist_depth"]),
                    force_history=batch["force_history"],
                ),
                "shuffle_depth": dict(
                    wrist_depth=_make_perturbed_tensor(
                        dataset,
                        "wrist_depth",
                        mode="shuffle",
                        base_tensor=batch["wrist_depth"],
                        perm_rows=depth_perm[idxs],
                    ),
                    force_history=batch["force_history"],
                ),
                "zero_force": dict(
                    wrist_depth=batch["wrist_depth"],
                    force_history=torch.zeros_like(batch["force_history"]),
                ),
                "shuffle_force": dict(
                    wrist_depth=batch["wrist_depth"],
                    force_history=_make_perturbed_tensor(
                        dataset,
                        "force_history",
                        mode="shuffle",
                        base_tensor=batch["force_history"],
                        perm_rows=force_perm[idxs],
                    ),
                ),
                "zero_both": dict(
                    wrist_depth=torch.zeros_like(batch["wrist_depth"]),
                    force_history=torch.zeros_like(batch["force_history"]),
                ),
                "shuffle_both": dict(
                    wrist_depth=_make_perturbed_tensor(
                        dataset,
                        "wrist_depth",
                        mode="shuffle",
                        base_tensor=batch["wrist_depth"],
                        perm_rows=depth_perm[idxs],
                    ),
                    force_history=_make_perturbed_tensor(
                        dataset,
                        "force_history",
                        mode="shuffle",
                        base_tensor=batch["force_history"],
                        perm_rows=force_perm[idxs],
                    ),
                ),
            }
            for key, spec in perturb_specs.items():
                pert_out = model(
                    front_rgb=batch["front_rgb"],
                    wrist_rgb=batch["wrist_rgb"],
                    wrist_depth=spec["wrist_depth"],
                    force_history=spec["force_history"],
                    proprio=batch["proprio"],
                    planner_base_action_local=batch["planner_base_action_local"],
                    proposal_actions_local=batch["proposal_actions_local"],
                    stage_token=batch.get("stage_token"),
                    contact_phase=batch.get("contact_phase"),
                    depth_proximity=batch.get("depth_proximity"),
                    gripper_state=batch.get("gripper_state"),
                )
                pert = _select_scores(
                    pert_out,
                    selection_mode=selection_mode,
                    w_safe=multi_utility_w_safe,
                    w_pareto=multi_utility_w_pareto,
                    w_yaw=multi_utility_w_yaw,
                    w_geom=multi_utility_w_geom,
                    w_risk=multi_utility_w_risk,
                ).detach().cpu().numpy()
                outputs_cache[key].append(pert)

    base_scores = np.concatenate(base_scores_all, axis=0)
    row_index = np.concatenate(row_index_all, axis=0)
    assert base_scores.shape[0] == n, (base_scores.shape, n)
    summary: dict[str, object] = {
        "rows": int(n),
        "proposal_count": int(base_scores.shape[1]),
        "row_index": row_index,
        "base_score_stats": {
            "mean": float(np.mean(base_scores)),
            "std": float(np.std(base_scores)),
            "min": float(np.min(base_scores)),
            "max": float(np.max(base_scores)),
        },
    }
    if sensitivity_audit:
        sensitivity: dict[str, object] = {}
        for key, chunks in outputs_cache.items():
            pert = np.concatenate(chunks, axis=0)
            sensitivity[key] = _candidatewise_delta_summary(base_scores, pert)
        summary["sensitivity"] = sensitivity
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_npz", required=True)
    ap.add_argument("--proposal_cache_npz", default="")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--output_json", required=True)
    ap.add_argument("--batch_size", type=int, default=64)
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
    device = torch.device(args.device)
    model, load_meta = _load_model(args.checkpoint, device=device)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    result = _evaluate(
        model,
        dataset,
        loader,
        device=device,
        sensitivity_audit=bool(args.sensitivity_audit),
        permutation_seed=int(args.permutation_seed),
        selection_mode=str(args.selection_mode),
        multi_utility_w_safe=float(args.multi_utility_w_safe),
        multi_utility_w_pareto=float(args.multi_utility_w_pareto),
        multi_utility_w_yaw=float(args.multi_utility_w_yaw),
        multi_utility_w_geom=float(args.multi_utility_w_geom),
        multi_utility_w_risk=float(args.multi_utility_w_risk),
    )
    report = {
        "dataset_npz": str(args.dataset_npz),
        "proposal_cache_npz": str(args.proposal_cache_npz),
        "checkpoint": str(args.checkpoint),
        "load_summary": _jsonable(load_meta["load_summary"]),
        "model_kwargs": _jsonable(load_meta["model_kwargs"]),
        "selection_mode": str(args.selection_mode),
        "multi_utility_weights": {
            "safe": float(args.multi_utility_w_safe),
            "pareto": float(args.multi_utility_w_pareto),
            "yaw": float(args.multi_utility_w_yaw),
            "geom": float(args.multi_utility_w_geom),
            "risk": float(args.multi_utility_w_risk),
        },
        "rows": result["rows"],
        "proposal_count": result["proposal_count"],
        "base_score_stats": _jsonable(result["base_score_stats"]),
    }
    if args.sensitivity_audit:
        report["sensitivity"] = _jsonable(result["sensitivity"])

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
