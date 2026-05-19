#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from prismatic.models.depth_force_local_proposal_policy import DepthForceLocalProposalPolicy
from prismatic.vla.datasets.depth_force_local_proposal_dataset import DepthForceLocalProposalDataset


def _load_eval_module():
    eval_path = Path(__file__).resolve().with_name("evaluate_depth_force_local_proposal_policy.py")
    spec = importlib.util.spec_from_file_location("depth_force_eval", eval_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load evaluation module from {eval_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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


def _load_model_from_checkpoint(checkpoint: str, *, device: torch.device) -> tuple[DepthForceLocalProposalPolicy, dict]:
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
    load_summary = {
        "missing_keys": sorted(missing),
        "unexpected_keys": sorted(unexpected),
        "loaded_keys": sorted(set(model.state_dict().keys()) & set(ckpt["model_state_dict"].keys())),
        "newly_initialized_keys": sorted(missing),
    }
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
    model = model.to(device)
    model.eval()
    return model, {"model_kwargs": model_kwargs, "load_summary": load_summary, "ckpt": ckpt}


def _group_summary(rows: np.ndarray, metrics: dict[str, np.ndarray]) -> dict[str, float]:
    idx = np.asarray(rows, dtype=np.int64)
    if idx.size == 0:
        return {"rows": 0}
    out = {"rows": int(idx.size)}
    for key, arr in metrics.items():
        arr_np = np.asarray(arr)
        if arr_np.ndim == 0:
            out[key] = float(arr_np.item())
            continue
        out[key] = float(np.mean(arr_np[idx]))
    return out


def _episode_summary(episodes: np.ndarray, metrics: dict[str, np.ndarray]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    episodes = np.asarray(episodes, dtype=np.int64).reshape(-1)
    for ep in np.unique(episodes):
        idx = np.where(episodes == int(ep))[0]
        out[str(int(ep))] = {
            "rows": int(idx.size),
            "selected_best_safe_hit_rate": float(np.mean(metrics["selected_best_safe_hit"][idx])),
            "selected_pareto_hit_rate": float(np.mean(metrics["selected_pareto_hit"][idx])),
            "selected_geom_gain_mean": float(np.mean(metrics["selected_geom_gain"][idx])),
            "selected_risk_delta_mean": float(np.mean(metrics["selected_risk_delta"][idx])),
            "best_safe_rank_mean": float(np.mean(metrics["best_safe_score_rank"][idx])),
            "pareto_rank_mean": float(np.mean(metrics["pareto_best_score_rank"][idx])),
            "yaw_match_rate": float(np.mean(metrics["selected_yaw_match_rate"][idx])),
            "correct_yaw_sign_rate": float(np.mean(metrics["selected_correct_yaw_sign"][idx])),
            "score_geom_gain_spearman_mean": float(np.nanmean(metrics["score_geom_gain_spearman"][idx])),
            "score_utility_spearman_mean": float(np.nanmean(metrics["score_utility_spearman"][idx])),
        }
    return out


def _oracle_selection_gap(dataset: DepthForceLocalProposalDataset, row_metrics: dict[str, np.ndarray]) -> dict[str, float]:
    geom_gain = np.asarray(dataset.data["proposal_geometry_gain"], dtype=np.float32)
    risk_delta = np.asarray(dataset.data["proposal_risk_delta"], dtype=np.float32)
    pareto_mask = np.asarray(dataset.data["proposal_pareto_mask"], dtype=np.float32) > 0.5
    best_safe_idx = np.asarray(dataset.data["proposal_best_safe_index"], dtype=np.int64)
    geom_best_idx = np.asarray(dataset.data["proposal_geom_top1_index"], dtype=np.int64)

    utility = geom_gain - np.maximum(risk_delta, 0.0)
    oracle_utility_idx = np.argmax(utility, axis=1)
    oracle_pareto_idx = np.zeros((geom_gain.shape[0],), dtype=np.int64)
    for i in range(geom_gain.shape[0]):
        if np.any(pareto_mask[i]):
            candidates = np.where(pareto_mask[i])[0]
            oracle_pareto_idx[i] = int(candidates[int(np.argmax(utility[i, candidates]))])
        else:
            oracle_pareto_idx[i] = int(best_safe_idx[i])

    def _summary(idx: np.ndarray) -> dict[str, float]:
        rows = np.arange(idx.size, dtype=np.int64)
        return {
            "geom_gain_mean": float(np.mean(geom_gain[rows, idx])),
            "risk_delta_mean": float(np.mean(risk_delta[rows, idx])),
        }

    selected_geom = np.asarray(row_metrics["selected_geom_gain"], dtype=np.float32)
    selected_risk = np.asarray(row_metrics["selected_risk_delta"], dtype=np.float32)
    selected_best_safe_hit = np.asarray(row_metrics["selected_best_safe_hit"], dtype=np.float32)
    selected_pareto_hit = np.asarray(row_metrics["selected_pareto_hit"], dtype=np.float32)

    best_safe_oracle = _summary(best_safe_idx)
    pareto_oracle = _summary(oracle_pareto_idx)
    geometry_oracle = _summary(geom_best_idx)
    utility_oracle = _summary(oracle_utility_idx)

    return {
        "selected_best_safe_hit_rate": float(np.mean(selected_best_safe_hit)),
        "selected_pareto_hit_rate": float(np.mean(selected_pareto_hit)),
        "selected_geom_gain_mean": float(np.mean(selected_geom)),
        "selected_risk_delta_mean": float(np.mean(selected_risk)),
        "oracle_best_safe_geom_gain_mean": best_safe_oracle["geom_gain_mean"],
        "oracle_best_safe_risk_delta_mean": best_safe_oracle["risk_delta_mean"],
        "oracle_pareto_geom_gain_mean": pareto_oracle["geom_gain_mean"],
        "oracle_pareto_risk_delta_mean": pareto_oracle["risk_delta_mean"],
        "oracle_geometry_geom_gain_mean": geometry_oracle["geom_gain_mean"],
        "oracle_geometry_risk_delta_mean": geometry_oracle["risk_delta_mean"],
        "oracle_utility_geom_gain_mean": utility_oracle["geom_gain_mean"],
        "oracle_utility_risk_delta_mean": utility_oracle["risk_delta_mean"],
        "gap_best_safe_geom_gain_mean": best_safe_oracle["geom_gain_mean"] - float(np.mean(selected_geom)),
        "gap_best_safe_risk_delta_mean": float(np.mean(selected_risk)) - best_safe_oracle["risk_delta_mean"],
        "gap_pareto_geom_gain_mean": pareto_oracle["geom_gain_mean"] - float(np.mean(selected_geom)),
        "gap_pareto_risk_delta_mean": float(np.mean(selected_risk)) - pareto_oracle["risk_delta_mean"],
        "gap_geometry_geom_gain_mean": geometry_oracle["geom_gain_mean"] - float(np.mean(selected_geom)),
        "gap_geometry_risk_delta_mean": float(np.mean(selected_risk)) - geometry_oracle["risk_delta_mean"],
        "gap_utility_geom_gain_mean": utility_oracle["geom_gain_mean"] - float(np.mean(selected_geom)),
        "gap_utility_risk_delta_mean": float(np.mean(selected_risk)) - utility_oracle["risk_delta_mean"],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_npz", required=True)
    ap.add_argument("--proposal_cache_npz", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--output_json", required=True)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--distance_tol", type=float, default=0.75)
    ap.add_argument("--yaw_presence_threshold", type=float, default=0.0025)
    ap.add_argument("--yaw_match_tol", type=float, default=0.0015)
    ap.add_argument(
        "--selection_mode",
        default="layered_multi",
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
    ap.add_argument("--batch_size_eval", type=int, default=32)
    ap.add_argument("--device", default="cuda" if False else "cpu")
    ap.add_argument("--sensitivity_audit", action="store_true")
    args = ap.parse_args()

    eval_mod = _load_eval_module()
    dataset = DepthForceLocalProposalDataset(args.dataset_npz, proposal_cache_npz=args.proposal_cache_npz)
    loader = DataLoader(dataset, batch_size=int(args.batch_size_eval), shuffle=False, num_workers=0, pin_memory=False)
    device = __import__("torch").device(args.device)
    model, load_meta = _load_model_from_checkpoint(args.checkpoint, device=device)
    metrics = eval_mod._evaluate_model(
        model,
        dataset,
        loader,
        device=device,
        distance_tol=float(args.distance_tol),
        yaw_presence_threshold=float(args.yaw_presence_threshold),
        yaw_match_tol=float(args.yaw_match_tol),
        sensitivity_audit=bool(args.sensitivity_audit),
        permutation_seed=0,
        selection_mode=str(args.selection_mode),
        multi_utility_w_safe=float(args.multi_utility_w_safe),
        multi_utility_w_pareto=float(args.multi_utility_w_pareto),
        multi_utility_w_yaw=float(args.multi_utility_w_yaw),
        multi_utility_w_geom=float(args.multi_utility_w_geom),
        multi_utility_w_risk=float(args.multi_utility_w_risk),
    )
    row_metrics = {k: v for k, v in metrics.items() if k != "sensitivity"}

    episodes = np.asarray(dataset.data["episode_index"], dtype=np.int64)
    yaw_aug = np.asarray(dataset.data.get("yaw_augmentation_applied", np.zeros((len(dataset),), dtype=np.float32)), dtype=np.float32) > 0.5
    yaw_opp = np.asarray(dataset.data.get("yaw_opportunity_label", np.zeros((len(dataset),), dtype=np.float32)), dtype=np.float32) > 0.5
    non_yaw = ~yaw_opp
    original = ~yaw_aug
    high_risk = np.max(np.asarray(dataset.data["proposal_risk_delta"], dtype=np.float32), axis=1) > 0.0
    weak_eps = {1, 8, 10, 19}
    strong_eps = {5, 16, 17, 20}
    weak_rows = np.isin(episodes, sorted(weak_eps))
    strong_rows = np.isin(episodes, sorted(strong_eps))

    per_episode = _episode_summary(episodes, row_metrics)
    groups = {
        "original_rows": _group_summary(np.where(original)[0], row_metrics),
        "yaw_augmented_rows": _group_summary(np.where(yaw_aug)[0], row_metrics),
        "yaw_opportunity_rows": _group_summary(np.where(yaw_opp)[0], row_metrics),
        "non_yaw_rows": _group_summary(np.where(non_yaw)[0], row_metrics),
        "high_risk_rows": _group_summary(np.where(high_risk)[0], row_metrics),
        "weak_episodes": _group_summary(np.where(weak_rows)[0], row_metrics),
        "strong_episodes": _group_summary(np.where(strong_rows)[0], row_metrics),
    }

    path_audit = {
        "pareto_pool_nonempty_rate": float(np.mean(row_metrics["pareto_pool_nonempty"])),
        "best_safe_in_pareto_pool_rate": float(np.mean(np.asarray(dataset.data["proposal_pareto_mask"], dtype=np.float32)[np.arange(len(dataset)), np.asarray(dataset.data["proposal_best_safe_index"], dtype=np.int64)] > 0.5)),
        "selected_from_pareto_pool_rate": float(np.mean(row_metrics["selected_from_pareto_pool"])),
        "fallback_rate": float(np.mean(row_metrics["fallback_used"])),
        "selected_by_best_safe_head_rate": float(np.mean(row_metrics["selected_by_best_safe_head"])),
        "selected_by_geometry_head_rate": float(np.mean(row_metrics["selected_by_geometry_head"])),
        "selected_with_yaw_bonus_rate": float(np.mean(row_metrics["selected_with_yaw_bonus"])),
        "risk_tiebreak_used_rate": float(np.mean(row_metrics["risk_tiebreak_used"])),
    }

    oracle_gap = _oracle_selection_gap(dataset, row_metrics)

    report = {
        "checkpoint": args.checkpoint,
        "selection_mode": args.selection_mode,
        "load_report": load_meta["load_summary"],
        "model_kwargs": _jsonable(load_meta["model_kwargs"]),
        "overall": {
            "selected_best_safe_hit_rate": float(np.mean(row_metrics["selected_best_safe_hit"])),
            "selected_pareto_hit_rate": float(np.mean(row_metrics["selected_pareto_hit"])),
            "selected_geom_gain_mean": float(np.mean(row_metrics["selected_geom_gain"])),
            "selected_risk_delta_mean": float(np.mean(row_metrics["selected_risk_delta"])),
            "best_safe_rank_mean": float(np.mean(row_metrics["best_safe_score_rank"])),
            "pareto_rank_mean": float(np.mean(row_metrics["pareto_best_score_rank"])),
            "selected_yaw_match_rate": float(np.mean(row_metrics["selected_yaw_match_rate"])),
            "selected_correct_yaw_sign_rate": float(np.mean(row_metrics["selected_correct_yaw_sign"])),
            "score_geom_gain_spearman_mean": float(np.nanmean(row_metrics["score_geom_gain_spearman"])),
            "score_utility_spearman_mean": float(np.nanmean(row_metrics["score_utility_spearman"])),
        },
        "per_episode": per_episode,
        "groups": groups,
        "path_audit": path_audit,
        "oracle_gap": oracle_gap,
    }
    if args.sensitivity_audit and "sensitivity" in metrics:
        report["sensitivity"] = {
            "zero_depth": {
                "mean_abs_score_delta": float(np.mean(metrics["sensitivity"]["zero_depth_mean_abs_score_delta"])),
                "max_abs_score_delta": float(np.mean(metrics["sensitivity"]["zero_depth_max_abs_score_delta"])),
                "argmax_changed_rate": float(np.mean(metrics["sensitivity"]["zero_depth_argmax_changed_rate"])),
                "rank_changed_rate": float(np.mean(metrics["sensitivity"]["zero_depth_rank_changed_rate"])),
            },
            "shuffle_depth": {
                "mean_abs_score_delta": float(np.mean(metrics["sensitivity"]["shuffle_depth_mean_abs_score_delta"])),
                "max_abs_score_delta": float(np.mean(metrics["sensitivity"]["shuffle_depth_max_abs_score_delta"])),
                "argmax_changed_rate": float(np.mean(metrics["sensitivity"]["shuffle_depth_argmax_changed_rate"])),
                "rank_changed_rate": float(np.mean(metrics["sensitivity"]["shuffle_depth_rank_changed_rate"])),
            },
            "zero_force": {
                "mean_abs_score_delta": float(np.mean(metrics["sensitivity"]["zero_force_mean_abs_score_delta"])),
                "max_abs_score_delta": float(np.mean(metrics["sensitivity"]["zero_force_max_abs_score_delta"])),
                "argmax_changed_rate": float(np.mean(metrics["sensitivity"]["zero_force_argmax_changed_rate"])),
                "rank_changed_rate": float(np.mean(metrics["sensitivity"]["zero_force_rank_changed_rate"])),
            },
            "shuffle_force": {
                "mean_abs_score_delta": float(np.mean(metrics["sensitivity"]["shuffle_force_mean_abs_score_delta"])),
                "max_abs_score_delta": float(np.mean(metrics["sensitivity"]["shuffle_force_max_abs_score_delta"])),
                "argmax_changed_rate": float(np.mean(metrics["sensitivity"]["shuffle_force_argmax_changed_rate"])),
                "rank_changed_rate": float(np.mean(metrics["sensitivity"]["shuffle_force_rank_changed_rate"])),
            },
            "zero_both": {
                "mean_abs_score_delta": float(np.mean(metrics["sensitivity"]["zero_both_mean_abs_score_delta"])),
                "max_abs_score_delta": float(np.mean(metrics["sensitivity"]["zero_both_max_abs_score_delta"])),
                "argmax_changed_rate": float(np.mean(metrics["sensitivity"]["zero_both_argmax_changed_rate"])),
                "rank_changed_rate": float(np.mean(metrics["sensitivity"]["zero_both_rank_changed_rate"])),
            },
            "shuffle_both": {
                "mean_abs_score_delta": float(np.mean(metrics["sensitivity"]["shuffle_both_mean_abs_score_delta"])),
                "max_abs_score_delta": float(np.mean(metrics["sensitivity"]["shuffle_both_max_abs_score_delta"])),
                "argmax_changed_rate": float(np.mean(metrics["sensitivity"]["shuffle_both_argmax_changed_rate"])),
                "rank_changed_rate": float(np.mean(metrics["sensitivity"]["shuffle_both_rank_changed_rate"])),
            },
        }

    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
