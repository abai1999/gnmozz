#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from prismatic.models.depth_force_local_proposal_policy import DepthForceLocalProposalPolicy
from prismatic.vla.datasets.depth_force_local_proposal_dataset import DepthForceLocalProposalDataset
from local_proposal_utils import evaluate_state_conditioned_proposals, select_best_indices


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


@torch.no_grad()
def _collect_cache(
    model: DepthForceLocalProposalPolicy,
    dataset: DepthForceLocalProposalDataset,
    loader: DataLoader,
    *,
    device: torch.device,
) -> dict[str, np.ndarray]:
    model.eval()
    arrays: dict[str, list[np.ndarray]] = {
        "row_index": [],
        "episode_index": [],
        "step_index": [],
        "proposal_actions_local": [],
        "proposal_scores_init": [],
        "proposal_geometry_cost": [],
        "proposal_risk_cost": [],
        "proposal_geometry_gain": [],
        "proposal_risk_delta": [],
        "proposal_pareto_mask": [],
        "proposal_budget_mask": [],
        "proposal_baseline_index": [],
        "proposal_geom_top1_index": [],
        "proposal_best_safe_index": [],
        "proposal_best_soft_index": [],
        "proposal_best_budget_index": [],
        "proposal_target_delta_local": [],
        "proposal_safe_target_action_local": [],
        "proposal_target_mode": [],
        "proposal_target_source": [],
        "current_pose_7d": [],
        "target_pose_7d": [],
        "planner_base_action_local": [],
        "proprio": [],
        "force_history": [],
        "wrist_depth": [],
        "front_rgb": [],
        "wrist_rgb": [],
        "gripper_state": [],
        "depth_proximity": [],
        "contact_phase": [],
        "stage_token": [],
    }
    for batch in loader:
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        outputs = model(
            front_rgb=batch["front_rgb"],
            wrist_rgb=batch["wrist_rgb"],
            wrist_depth=batch["wrist_depth"],
            force_history=batch["force_history"],
            proprio=batch["proprio"],
            planner_base_action_local=batch["planner_base_action_local"],
            stage_token=batch.get("stage_token"),
            contact_phase=batch.get("contact_phase"),
            depth_proximity=batch.get("depth_proximity"),
            gripper_state=batch.get("gripper_state"),
        )
        bsz = int(batch["row_index"].shape[0])
        for i in range(bsz):
            row_idx = int(batch["row_index"][i].detach().cpu().item())
            current_pose = np.asarray(batch["current_pose_7d"][i].detach().cpu().numpy(), dtype=np.float32)[None, :]
            target_pose = np.asarray(batch["target_pose_7d"][i].detach().cpu().numpy(), dtype=np.float32)[None, :]
            cand = outputs["proposal_actions_local"][i].detach().cpu().numpy()[None, :, :]
            contact = np.asarray([float(dataset.data.get("contact_label", np.zeros((len(dataset),), dtype=np.float32))[row_idx])], dtype=np.float32)
            force_spike = np.asarray([float(dataset.data.get("force_spike_label", np.zeros((len(dataset),), dtype=np.float32))[row_idx])], dtype=np.float32)
            jam = np.asarray([float(dataset.data.get("jam_label", np.zeros((len(dataset),), dtype=np.float32))[row_idx])], dtype=np.float32)
            motion_stall = np.asarray([float(dataset.data.get("motion_stall_label", np.zeros((len(dataset),), dtype=np.float32))[row_idx])], dtype=np.float32)
            near_depth = np.asarray([float(dataset.data.get("near_depth_label", np.zeros((len(dataset),), dtype=np.float32))[row_idx])], dtype=np.float32)
            kin_invalid = np.asarray([float(dataset.data.get("kinematic_invalid_label", np.zeros((len(dataset),), dtype=np.float32))[row_idx])], dtype=np.float32)
            action_range_invalid = np.asarray([float(dataset.data.get("action_range_invalid_label", np.zeros((len(dataset),), dtype=np.float32))[row_idx])], dtype=np.float32)
            gripper_state = np.asarray([float(batch["gripper_state"][i].detach().cpu().item())], dtype=np.float32)
            eval_out = evaluate_state_conditioned_proposals(
                current_pose=current_pose,
                target_pose=target_pose,
                candidate_actions=cand,
                contact=contact,
                force_spike=force_spike,
                jam=jam,
                motion_stall=motion_stall,
                near_depth=near_depth,
                kin_invalid=kin_invalid,
                action_range_invalid=action_range_invalid,
                gripper_state=gripper_state,
            )
            geom = np.asarray(eval_out["candidate_geometry_cost"], dtype=np.float32)[0]
            risk = np.asarray(eval_out["candidate_risk_cost"], dtype=np.float32)[0]
            base_idx = int(batch["proposal_baseline_index"][i].detach().cpu().item())
            bests = select_best_indices(
                geom[None, :],
                risk[None, :],
                baseline_index=np.asarray([base_idx], dtype=np.int64),
                geo_margin=0.0,
                risk_budget=0.03,
                soft_alpha=0.3,
            )
            arrays["row_index"].append(batch["row_index"][i].detach().cpu().numpy()[None])
            arrays["episode_index"].append(batch["episode_index"][i].detach().cpu().numpy()[None])
            arrays["step_index"].append(batch["step_index"][i].detach().cpu().numpy()[None])
            arrays["proposal_actions_local"].append(outputs["proposal_actions_local"][i].detach().cpu().numpy()[None])
            arrays["proposal_scores_init"].append(outputs["proposal_scores"][i].detach().cpu().numpy()[None])
            arrays["proposal_geometry_cost"].append(geom[None, :])
            arrays["proposal_risk_cost"].append(risk[None, :])
            arrays["proposal_geometry_gain"].append((geom[base_idx] - geom)[None, :])
            arrays["proposal_risk_delta"].append((risk - risk[base_idx])[None, :])
            arrays["proposal_pareto_mask"].append(bests["safe_mask"].astype(np.float32))
            arrays["proposal_budget_mask"].append(bests["budget_mask"].astype(np.float32))
            arrays["proposal_baseline_index"].append(bests["baseline_index"].astype(np.int64))
            arrays["proposal_geom_top1_index"].append(bests["geom_top1_index"].astype(np.int64))
            arrays["proposal_best_safe_index"].append(bests["best_safe_index"].astype(np.int64))
            arrays["proposal_best_soft_index"].append(bests["best_soft_index"].astype(np.int64))
            arrays["proposal_best_budget_index"].append(bests["best_budget_index"].astype(np.int64))
            arrays["proposal_target_delta_local"].append(batch["proposal_target_delta_local"][i].detach().cpu().numpy()[None])
            arrays["proposal_safe_target_action_local"].append(batch["proposal_safe_target_action_local"][i].detach().cpu().numpy()[None])
            arrays["proposal_target_mode"].append(np.asarray([batch["proposal_target_mode"][i]], dtype="U32"))
            arrays["proposal_target_source"].append(np.asarray([batch["proposal_target_source"][i]], dtype="U64"))
            arrays["current_pose_7d"].append(batch["current_pose_7d"][i].detach().cpu().numpy()[None])
            arrays["target_pose_7d"].append(batch["target_pose_7d"][i].detach().cpu().numpy()[None])
            arrays["planner_base_action_local"].append(batch["planner_base_action_local"][i].detach().cpu().numpy()[None])
            arrays["proprio"].append(batch["proprio"][i].detach().cpu().numpy()[None])
            arrays["force_history"].append(batch["force_history"][i].detach().cpu().numpy()[None])
            arrays["wrist_depth"].append(batch["wrist_depth"][i].detach().cpu().numpy()[None])
            arrays["front_rgb"].append(batch["front_rgb"][i].detach().cpu().numpy()[None])
            arrays["wrist_rgb"].append(batch["wrist_rgb"][i].detach().cpu().numpy()[None])
            arrays["gripper_state"].append(batch["gripper_state"][i].detach().cpu().numpy()[None])
            arrays["depth_proximity"].append(batch["depth_proximity"][i].detach().cpu().numpy()[None])
            arrays["contact_phase"].append(batch["contact_phase"][i].detach().cpu().numpy()[None])
            arrays["stage_token"].append(batch["stage_token"][i].detach().cpu().numpy()[None])

    out = {}
    for key, chunks in arrays.items():
        if not chunks:
            continue
        out[key] = np.concatenate(chunks, axis=0)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_npz", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--output_npz", required=True)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    dataset = DepthForceLocalProposalDataset(args.dataset_npz)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    inferred_hidden = int(ckpt["model_state_dict"]["proposal_head.0.weight"].shape[0])
    model_kwargs = dict(ckpt.get("model_kwargs", {}))
    model = DepthForceLocalProposalPolicy(
        proposal_count=int(ckpt.get("proposal_count", 8)),
        state_dim=int(ckpt.get("state_dim", 384)),
        hidden_dim=int(ckpt.get("hidden_dim", inferred_hidden)),
        **model_kwargs,
    )
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    device = torch.device(args.device)
    model = model.to(device)

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    cache = _collect_cache(model, dataset, loader, device=device)
    out_path = Path(args.output_npz)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **cache)

    report = {
        "dataset_npz": str(args.dataset_npz),
        "checkpoint": str(args.checkpoint),
        "output_npz": str(out_path),
        "rows": int(cache["row_index"].shape[0]) if "row_index" in cache else 0,
        "proposal_count": int(cache["proposal_actions_local"].shape[1]) if "proposal_actions_local" in cache else 0,
        "model_kwargs": _jsonable(model_kwargs),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
