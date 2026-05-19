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


def _find_rows_for_episodes(dataset: DepthForceLocalProposalDataset, episodes: list[int]) -> np.ndarray:
    eps = np.asarray(dataset.data["episode_index"], dtype=np.int64)
    return np.where(np.isin(eps, np.asarray(episodes, dtype=np.int64)))[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_npz", required=True)
    ap.add_argument("--proposal_cache_npz", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--trace_dir", required=True)
    ap.add_argument("--episodes", default="1,5,8,10,16,17,19,20")
    ap.add_argument("--selection_mode", default="layered_multi")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--distance_tol", type=float, default=0.75)
    ap.add_argument("--yaw_presence_threshold", type=float, default=0.0025)
    ap.add_argument("--yaw_match_tol", type=float, default=0.0015)
    ap.add_argument("--close_contact_depth_threshold", type=float, default=0.08)
    ap.add_argument("--close_contact_force_threshold", type=float, default=0.05)
    ap.add_argument("--shadow_name", default="depth_force_local_proposal_shadow_fixed_8ep")
    args = ap.parse_args()

    eval_mod = _load_eval_module()
    dataset = DepthForceLocalProposalDataset(args.dataset_npz, proposal_cache_npz=args.proposal_cache_npz)
    loader = DataLoader(dataset, batch_size=int(args.batch_size), shuffle=False, num_workers=0, pin_memory=False)
    device = torch.device(args.device)
    model, load_meta = _load_model_from_checkpoint(args.checkpoint, device=device)
    metrics = eval_mod._evaluate_model(
        model,
        dataset,
        loader,
        device=device,
        distance_tol=float(args.distance_tol),
        yaw_presence_threshold=float(args.yaw_presence_threshold),
        yaw_match_tol=float(args.yaw_match_tol),
        sensitivity_audit=False,
        permutation_seed=0,
        selection_mode=str(args.selection_mode),
    )

    episodes = [int(x) for x in str(args.episodes).split(",") if x.strip()]
    row_sel = _find_rows_for_episodes(dataset, episodes)
    trace_root = Path(args.trace_dir)
    trace_dir = trace_root / "gripper_traces"
    trace_dir.mkdir(parents=True, exist_ok=True)

    eps = np.asarray(dataset.data["episode_index"], dtype=np.int64)
    steps = np.asarray(dataset.data.get("step_index", np.arange(len(dataset), dtype=np.int64)), dtype=np.int64)
    depth_prox = np.asarray(dataset.data.get("depth_proximity", np.zeros((len(dataset),), dtype=np.float32)), dtype=np.float32)
    contact_phase = np.asarray(dataset.data.get("contact_state", np.zeros((len(dataset),), dtype=np.int64)), dtype=np.int64)
    gripper_state = np.asarray(dataset.data.get("gripper_state", np.zeros((len(dataset),), dtype=np.float32)), dtype=np.float32)
    baseline_idx = np.asarray(dataset.data["proposal_baseline_index"], dtype=np.int64)
    best_safe_idx = np.asarray(dataset.data["proposal_best_safe_index"], dtype=np.int64)
    best_geom_idx = np.asarray(dataset.data["proposal_geom_top1_index"], dtype=np.int64)
    selected_idx = np.asarray(metrics["selected_idx"], dtype=np.int64)
    selected_from_pareto_pool = np.asarray(metrics["selected_from_pareto_pool"], dtype=np.float32)
    fallback_used = np.asarray(metrics["fallback_used"], dtype=np.float32)
    selected_by_best_safe_head = np.asarray(metrics["selected_by_best_safe_head"], dtype=np.float32)
    selected_by_geometry_head = np.asarray(metrics["selected_by_geometry_head"], dtype=np.float32)
    selected_with_yaw_bonus = np.asarray(metrics["selected_with_yaw_bonus"], dtype=np.float32)
    risk_tiebreak_used = np.asarray(metrics["risk_tiebreak_used"], dtype=np.float32)
    selected_geom_gain = np.asarray(metrics["selected_geom_gain"], dtype=np.float32)
    selected_risk_delta = np.asarray(metrics["selected_risk_delta"], dtype=np.float32)
    selected_best_safe_hit = np.asarray(metrics["selected_best_safe_hit"], dtype=np.float32)
    selected_pareto_hit = np.asarray(metrics["selected_pareto_hit"], dtype=np.float32)
    selected_yaw_match_rate = np.asarray(metrics["selected_yaw_match_rate"], dtype=np.float32)
    selected_correct_yaw_sign = np.asarray(metrics["selected_correct_yaw_sign"], dtype=np.float32)
    best_safe_score_rank = np.asarray(metrics["best_safe_score_rank"], dtype=np.float32)
    pareto_best_score_rank = np.asarray(metrics["pareto_best_score_rank"], dtype=np.float32)
    selected_score = np.asarray(metrics["selected_score"], dtype=np.float32)
    set_min_dist_best_safe = np.asarray(metrics["set_min_dist_best_safe"], dtype=np.float32)
    set_min_dist_best_geom = np.asarray(metrics["set_min_dist_best_geom"], dtype=np.float32)

    records: list[dict[str, object]] = []
    for row_idx in row_sel:
        ep = int(eps[row_idx])
        step = int(steps[row_idx])
        trace = {
            "episode": ep,
            "step": step,
            "row_index": int(row_idx),
            "shadow_name": str(args.shadow_name),
            "shadow_gate_open": True,
            "shadow_applied": False,
            "shadow_changed": bool(int(selected_idx[row_idx]) != int(baseline_idx[row_idx])),
            "shadow_close_contact": bool(float(depth_prox[row_idx]) <= float(args.close_contact_depth_threshold) or int(contact_phase[row_idx]) > 0),
            "shadow_high_force": bool(np.linalg.norm(np.asarray(dataset.data.get("force_history", dataset.data.get("ft_hist"))[row_idx], dtype=np.float32)[-1, :3]) >= float(args.close_contact_force_threshold)) if ("force_history" in dataset.data or "ft_hist" in dataset.data) else False,
            "shadow_selected_idx": int(selected_idx[row_idx]),
            "shadow_baseline_idx": int(baseline_idx[row_idx]),
            "shadow_best_safe_idx": int(best_safe_idx[row_idx]),
            "shadow_best_geom_idx": int(best_geom_idx[row_idx]),
            "shadow_selected_from_pareto_pool": bool(selected_from_pareto_pool[row_idx] > 0.5),
            "shadow_fallback_used": bool(fallback_used[row_idx] > 0.5),
            "shadow_selected_by_best_safe_head": bool(selected_by_best_safe_head[row_idx] > 0.5),
            "shadow_selected_by_geometry_head": bool(selected_by_geometry_head[row_idx] > 0.5),
            "shadow_selected_with_yaw_bonus": bool(selected_with_yaw_bonus[row_idx] > 0.5),
            "shadow_risk_tiebreak_used": bool(risk_tiebreak_used[row_idx] > 0.5),
            "shadow_selected_geom_gain": float(selected_geom_gain[row_idx]),
            "shadow_selected_risk_delta": float(selected_risk_delta[row_idx]),
            "shadow_selected_best_safe_hit": bool(selected_best_safe_hit[row_idx] > 0.5),
            "shadow_selected_pareto_hit": bool(selected_pareto_hit[row_idx] > 0.5),
            "shadow_selected_yaw_match_rate": bool(selected_yaw_match_rate[row_idx] > 0.5),
            "shadow_selected_correct_yaw_sign": bool(selected_correct_yaw_sign[row_idx] > 0.5),
            "shadow_best_safe_score_rank": float(best_safe_score_rank[row_idx]),
            "shadow_pareto_score_rank": float(pareto_best_score_rank[row_idx]),
            "shadow_selected_score": float(selected_score[row_idx]),
            "shadow_set_min_dist_best_safe": float(set_min_dist_best_safe[row_idx]),
            "shadow_set_min_dist_best_geom": float(set_min_dist_best_geom[row_idx]),
            "shadow_depth_proximity": float(depth_prox[row_idx]),
            "shadow_contact_phase": int(contact_phase[row_idx]),
            "shadow_gripper_state": float(gripper_state[row_idx]),
            "shadow_non_invasive": True,
            "shadow_applied_index": int(baseline_idx[row_idx]),
            "shadow_applied_action_local": np.asarray(dataset.data["proposal_actions_local"][row_idx, baseline_idx[row_idx]], dtype=np.float32),
            "shadow_selected_action_local": np.asarray(dataset.data["proposal_actions_local"][row_idx, selected_idx[row_idx]], dtype=np.float32),
            "shadow_best_safe_action_local": np.asarray(dataset.data["proposal_actions_local"][row_idx, best_safe_idx[row_idx]], dtype=np.float32),
            "shadow_best_geom_action_local": np.asarray(dataset.data["proposal_actions_local"][row_idx, best_geom_idx[row_idx]], dtype=np.float32),
            "shadow_candidate_count": int(np.asarray(dataset.data["proposal_actions_local"][row_idx]).shape[0]),
            "shadow_selected_candidate_mask": np.asarray(dataset.data.get("proposal_pareto_mask", np.ones_like(dataset.data["proposal_actions_local"][row_idx, :, 0])), dtype=np.float32)[row_idx] if False else None,
        }
        # Remove the placeholder above to keep the JSON light and robust.
        trace.pop("shadow_selected_candidate_mask", None)
        records.append(trace)

    out_files = []
    by_episode: dict[int, list[dict[str, object]]] = {}
    for rec in records:
        by_episode.setdefault(int(rec["episode"]), []).append(rec)
    for ep, rows in sorted(by_episode.items()):
        out_path = trace_dir / f"ep{ep:03d}_gripper_trace.jsonl"
        out_path.write_text("\n".join(json.dumps(_jsonable(r)) for r in rows) + "\n", encoding="utf-8")
        out_files.append(str(out_path))

    manifest = {
        "shadow_name": args.shadow_name,
        "checkpoint": str(args.checkpoint),
        "dataset_npz": str(args.dataset_npz),
        "proposal_cache_npz": str(args.proposal_cache_npz),
        "episodes": episodes,
        "selection_mode": str(args.selection_mode),
        "distance_tol": float(args.distance_tol),
        "yaw_presence_threshold": float(args.yaw_presence_threshold),
        "yaw_match_tol": float(args.yaw_match_tol),
        "close_contact_depth_threshold": float(args.close_contact_depth_threshold),
        "close_contact_force_threshold": float(args.close_contact_force_threshold),
        "rows": int(len(records)),
        "trace_files": out_files,
        "model_kwargs": _jsonable(load_meta["model_kwargs"]),
        "load_summary": _jsonable(load_meta["load_summary"]),
        "selected_best_safe_hit_rate": float(np.mean(selected_best_safe_hit[row_sel])) if row_sel.size else 0.0,
        "selected_pareto_hit_rate": float(np.mean(selected_pareto_hit[row_sel])) if row_sel.size else 0.0,
        "selected_geom_gain_mean": float(np.mean(selected_geom_gain[row_sel])) if row_sel.size else 0.0,
        "selected_risk_delta_mean": float(np.mean(selected_risk_delta[row_sel])) if row_sel.size else 0.0,
        "best_safe_rank_mean": float(np.mean(best_safe_score_rank[row_sel])) if row_sel.size else 0.0,
        "pareto_rank_mean": float(np.mean(pareto_best_score_rank[row_sel])) if row_sel.size else 0.0,
    }
    (trace_root / "shadow_manifest.json").write_text(json.dumps(_jsonable(manifest), indent=2), encoding="utf-8")
    print(json.dumps(_jsonable(manifest), indent=2))


if __name__ == "__main__":
    main()
