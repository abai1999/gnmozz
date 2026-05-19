#!/usr/bin/env python3
"""Offline comparison: Target-Conditioned Alignment v2 vs final_full shadow.

Runs both models on the v2 dataset rows and compares candidate selections
against oracle best_stage_action_index.  Pure offline — no runtime changes.
"""
from __future__ import annotations

import argparse, json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from prismatic.models.target_conditioned_alignment_policy import TargetConditionedAlignmentPolicy
from prismatic.models.depth_force_local_proposal_policy import DepthForceLocalProposalPolicy
from prismatic.vla.datasets.target_conditioned_alignment_v2_dataset import TargetConditionedAlignmentV2Dataset


def _load_v2(ckpt_path: str, device: torch.device) -> TargetConditionedAlignmentPolicy:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = TargetConditionedAlignmentPolicy(proposal_count=8).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()
    return model


def _load_final_full(ckpt_path: str, device: torch.device) -> DepthForceLocalProposalPolicy:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    hidden_dim = int(ckpt.get("hidden_dim", ckpt["model_state_dict"]["proposal_head.0.weight"].shape[0]))
    model = DepthForceLocalProposalPolicy(
        proposal_count=8, state_dim=int(ckpt.get("state_dim", 384)), hidden_dim=hidden_dim,
    )
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.to(device).eval()
    return model


def _select_layered_multi_head_scores(multi, controller=None):
    """Replicate the layered selection logic from StageAwareRefiner."""
    best_safe = multi[..., 0]
    pareto = multi[..., 1]
    yaw_match = multi[..., 2]
    risk_safe = multi[..., 3]
    geom_gain = multi[..., 4]
    pareto_ok = (pareto >= 0.0).float()
    yaw_bonus = torch.where(yaw_match > 0.3, 0.15, 0.0)
    score = best_safe + pareto_ok * 0.3 + yaw_bonus
    return score


def _run_final_full(model, batch, device):
    """Run final_full on a batch and return selected indices + scores."""
    with torch.no_grad():
        fr = batch.get("front_rgb", torch.zeros(1, 3, 128, 128)).to(device)
        wr = batch.get("wrist_rgb", torch.zeros(1, 3, 128, 128)).to(device)
        wd = batch["wrist_depth"].to(device)
        fh = batch["force_history"].to(device)
        pr = batch["proprio"].to(device)
        ba = batch["planner_base_action_local"].to(device)

        # final_full needs specific input shapes
        if wd.ndim == 3:
            wd = wd.unsqueeze(1)
        if fr.ndim == 5:
            fr = fr.squeeze(1)
        if wr.ndim == 5:
            wr = wr.squeeze(1)

        # Call final_full forward
        bsz = ba.shape[0]
        stage_token = torch.zeros(bsz, dtype=torch.long, device=device)
        contact_phase = torch.zeros(bsz, dtype=torch.long, device=device)
        depth_proximity = torch.zeros(bsz, device=device)
        gripper_state = torch.zeros(bsz, device=device)

        out = model(fr, wr, wd, fh, pr, ba,
                    proposal_actions_local=None,
                    stage_token=stage_token,
                    contact_phase=contact_phase,
                    depth_proximity=depth_proximity,
                    gripper_state=gripper_state)

        multi = out["multi_head_scores"]
        scores = _select_layered_multi_head_scores(multi)
        selected_idx = scores.argmax(dim=-1)
        return selected_idx.cpu(), scores.cpu()


def _run_v2(model, batch, device):
    """Run v2 on a batch and return selected indices + scores."""
    with torch.no_grad():
        wd = batch["wrist_depth"].to(device)
        fh = batch["force_history"].to(device)
        pr = batch["proprio"].to(device)
        ba = batch["planner_base_action_local"].to(device)
        td = batch["current_to_target_delta_local"].to(device)
        pa = batch["proposal_actions"].to(device)
        pd = batch["post_candidate_delta"].to(device)
        xi = batch["xy_improvement"].to(device)
        zi = batch["z_improvement"].to(device)
        yi = batch["yaw_improvement"].to(device)
        gi = batch["geometry_improvement"].to(device)

        out = model(wd, fh, pr, ba, td, pa, pd, xi, zi, yi, gi)
        scores = out["candidate_scores"]
        selected_idx = scores.argmax(dim=-1)
        return selected_idx.cpu(), scores.cpu()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--v2_ckpt", type=Path, required=True)
    parser.add_argument("--final_full_ckpt", type=Path, required=True)
    parser.add_argument("--dataset_npz", type=Path, required=True)
    parser.add_argument("--output_report", type=Path, required=True)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"[audit] loading models on {device}...")

    v2 = _load_v2(str(args.v2_ckpt), device)
    ff = _load_final_full(str(args.final_full_ckpt), device)
    print(f"[audit] v2 params={sum(p.numel() for p in v2.parameters()):,}  "
          f"ff params={sum(p.numel() for p in ff.parameters()):,}")

    ds = TargetConditionedAlignmentV2Dataset(str(args.dataset_npz), stage_bucket_filter=None)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)
    print(f"[audit] dataset rows={len(ds)}")

    # Collect results
    results = defaultdict(list)

    for batch in loader:
        bsz = len(batch["row_index"])
        oracle_idx = batch["best_stage_action_index"]
        buckets = batch["stage_bucket"]
        row_indices = batch["row_index"].tolist()
        ep_indices = [str(x) for x in batch.get("episode_index", ["?"] * bsz)]

        # V2 inference
        v2_idx, v2_scores = _run_v2(v2, batch, device)
        # Final_full inference
        ff_idx, ff_scores = _run_final_full(ff, batch, device)

        for j in range(bsz):
            b = str(buckets[j])
            o = int(oracle_idx[j].item())
            vi = int(v2_idx[j].item())
            fi = int(ff_idx[j].item())

            results["bucket"].append(b)
            results["row_index"].append(int(row_indices[j]))
            results["episode"].append(ep_indices[j])
            results["oracle_idx"].append(o)
            results["v2_idx"].append(vi)
            results["ff_idx"].append(fi)
            results["v2_oracle_match"].append(int(vi == o))
            results["ff_oracle_match"].append(int(fi == o))
            results["v2_ff_agree"].append(int(vi == fi))

            # Post-error of selected candidates
            post_xy = batch["post_xy_error"][j]
            post_z = batch["post_z_error"][j]
            post_yaw = batch["post_yaw_error"][j]
            geom_imp = batch["geometry_improvement"][j]
            overshoot_xy = batch["overshoot_xy"][j]
            overshoot_z = batch["overshoot_z"][j]

            results["v2_post_xy"].append(float(post_xy[vi].item()))
            results["v2_post_z"].append(float(post_z[vi].item()))
            results["v2_post_yaw"].append(float(post_yaw[vi].item()))
            results["v2_geom_imp"].append(float(geom_imp[vi].item()))
            results["v2_overshoot_xy"].append(float(overshoot_xy[vi].item()))
            results["v2_overshoot_z"].append(float(overshoot_z[vi].item()))

            results["ff_post_xy"].append(float(post_xy[fi].item()))
            results["ff_post_z"].append(float(post_z[fi].item()))
            results["ff_post_yaw"].append(float(post_yaw[fi].item()))
            results["ff_geom_imp"].append(float(geom_imp[fi].item()))
            results["ff_overshoot_xy"].append(float(overshoot_xy[fi].item()))
            results["ff_overshoot_z"].append(float(overshoot_z[fi].item()))

            results["oracle_post_xy"].append(float(post_xy[o].item()))
            results["oracle_post_z"].append(float(post_z[o].item()))
            results["oracle_post_yaw"].append(float(post_yaw[o].item()))
            results["oracle_geom_imp"].append(float(geom_imp[o].item()))

            # current error
            results["current_xy"].append(float(batch["current_xy_error"][j].item()))
            results["current_z"].append(float(batch["current_z_error"][j].item()))
            results["current_yaw"].append(float(batch["current_yaw_error"][j].item()))

    # --- Per-bucket aggregates ---
    buckets_order = ["micro_contact_refine", "near_alignment", "mid_approach_assist", "far_coarse_approach"]
    per_bucket = {}
    for b in buckets_order:
        indices = [i for i, bb in enumerate(results["bucket"]) if bb == b]
        if not indices:
            continue
        n = len(indices)
        per_bucket[b] = {
            "n": n,
            "v2_top1": float(np.mean([results["v2_oracle_match"][i] for i in indices])),
            "ff_top1": float(np.mean([results["ff_oracle_match"][i] for i in indices])),
            "v2_ff_agree": float(np.mean([results["v2_ff_agree"][i] for i in indices])),
            "v2_post_xy_mean": float(np.mean([results["v2_post_xy"][i] for i in indices])),
            "ff_post_xy_mean": float(np.mean([results["ff_post_xy"][i] for i in indices])),
            "oracle_post_xy_mean": float(np.mean([results["oracle_post_xy"][i] for i in indices])),
            "v2_post_z_mean": float(np.mean([results["v2_post_z"][i] for i in indices])),
            "ff_post_z_mean": float(np.mean([results["ff_post_z"][i] for i in indices])),
            "oracle_post_z_mean": float(np.mean([results["oracle_post_z"][i] for i in indices])),
            "v2_post_yaw_mean": float(np.mean([results["v2_post_yaw"][i] for i in indices])),
            "ff_post_yaw_mean": float(np.mean([results["ff_post_yaw"][i] for i in indices])),
            "oracle_post_yaw_mean": float(np.mean([results["oracle_post_yaw"][i] for i in indices])),
            "v2_geom_imp_mean": float(np.mean([results["v2_geom_imp"][i] for i in indices])),
            "ff_geom_imp_mean": float(np.mean([results["ff_geom_imp"][i] for i in indices])),
            "oracle_geom_imp_mean": float(np.mean([results["oracle_geom_imp"][i] for i in indices])),
            "v2_overshoot_xy_rate": float(np.mean([results["v2_overshoot_xy"][i] for i in indices])),
            "ff_overshoot_xy_rate": float(np.mean([results["ff_overshoot_xy"][i] for i in indices])),
            "v2_overshoot_z_rate": float(np.mean([results["v2_overshoot_z"][i] for i in indices])),
            "ff_overshoot_z_rate": float(np.mean([results["ff_overshoot_z"][i] for i in indices])),
            "current_xy_mean": float(np.mean([results["current_xy"][i] for i in indices])),
            "current_z_mean": float(np.mean([results["current_z"][i] for i in indices])),
            "current_yaw_mean": float(np.mean([results["current_yaw"][i] for i in indices])),
        }

    # --- Candidate histograms ---
    v2_hist = dict(Counter(results["v2_idx"]))
    ff_hist = dict(Counter(results["ff_idx"]))
    oracle_hist = dict(Counter(results["oracle_idx"]))

    # --- Overall ---
    overall = {
        "n": len(results["row_index"]),
        "v2_top1": float(np.mean(results["v2_oracle_match"])),
        "ff_top1": float(np.mean(results["ff_oracle_match"])),
        "v2_ff_agree": float(np.mean(results["v2_ff_agree"])),
        "v2_post_xy_mean": float(np.mean(results["v2_post_xy"])),
        "ff_post_xy_mean": float(np.mean(results["ff_post_xy"])),
        "oracle_post_xy_mean": float(np.mean(results["oracle_post_xy"])),
        "v2_post_z_mean": float(np.mean(results["v2_post_z"])),
        "ff_post_z_mean": float(np.mean(results["ff_post_z"])),
        "oracle_post_z_mean": float(np.mean(results["oracle_post_z"])),
        "v2_geom_imp_mean": float(np.mean(results["v2_geom_imp"])),
        "ff_geom_imp_mean": float(np.mean(results["ff_geom_imp"])),
        "oracle_geom_imp_mean": float(np.mean(results["oracle_geom_imp"])),
    }

    report = {
        "audit": "v2_vs_final_full_shadow",
        "v2_ckpt": str(args.v2_ckpt),
        "final_full_ckpt": str(args.final_full_ckpt),
        "dataset_npz": str(args.dataset_npz),
        "overall": overall,
        "per_bucket": per_bucket,
        "candidate_histograms": {
            "v2": v2_hist,
            "final_full": ff_hist,
            "oracle": oracle_hist,
        },
    }

    # --- Near/micro verdict ---
    near_micro_n = per_bucket.get("near_alignment", {}).get("n", 0) + per_bucket.get("micro_contact_refine", {}).get("n", 0)
    near_v2_top1 = per_bucket.get("near_alignment", {}).get("v2_top1", 0)
    near_ff_top1 = per_bucket.get("near_alignment", {}).get("ff_top1", 0)
    v2_better_near = near_v2_top1 > near_ff_top1 + 0.05

    far_v2_top1 = per_bucket.get("far_coarse_approach", {}).get("v2_top1", 0)
    far_ff_top1 = per_bucket.get("far_coarse_approach", {}).get("ff_top1", 0)

    report["verdict"] = {
        "v2_better_than_ff_in_near_micro": bool(v2_better_near),
        "v2_near_top1": near_v2_top1,
        "ff_near_top1": near_ff_top1,
        "v2_far_top1": far_v2_top1,
        "ff_far_top1": far_ff_top1,
        "far_bucket_should_not_be_enabled": bool(far_v2_top1 < 0.7),
        "v2_ready_for_near_micro_shadow": bool(v2_better_near and near_micro_n > 100),
        "next_step": (
            "near_zone_gated_runtime_assist_shadow"
            if v2_better_near and near_micro_n > 100
            else "improve_model_or_data"
        ),
    }

    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_report, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)

    # Print summary
    print(f"\n=== OVERALL (n={overall['n']}) ===")
    print(f"  v2 top1={overall['v2_top1']:.4f}  ff top1={overall['ff_top1']:.4f}  v2-ff agree={overall['v2_ff_agree']:.4f}")
    print(f"  v2 post_z={overall['v2_post_z_mean']:.5f}  ff post_z={overall['ff_post_z_mean']:.5f}  oracle post_z={overall['oracle_post_z_mean']:.5f}")
    print(f"  v2 geom_imp={overall['v2_geom_imp_mean']:.5f}  ff geom_imp={overall['ff_geom_imp_mean']:.5f}  oracle geom_imp={overall['oracle_geom_imp_mean']:.5f}")

    print(f"\n=== PER BUCKET ===")
    header = f"{'bucket':<25} {'n':>5} {'v2_top1':>8} {'ff_top1':>8} {'v2_post_z':>10} {'ff_post_z':>10} {'v2_overshoot_z':>14}"
    print(header)
    print("-" * len(header))
    for b in buckets_order:
        s = per_bucket.get(b, {})
        if not s:
            continue
        print(f"{b:<25} {s['n']:>5} {s['v2_top1']:>8.4f} {s['ff_top1']:>8.4f} "
              f"{s['v2_post_z_mean']:>10.5f} {s['ff_post_z_mean']:>10.5f} {s['v2_overshoot_z_rate']:>14.4f}")

    print(f"\n=== CANDIDATE HISTOGRAMS ===")
    print(f"  v2:     {dict(sorted(v2_hist.items()))}")
    print(f"  ff:     {dict(sorted(ff_hist.items()))}")
    print(f"  oracle: {dict(sorted(oracle_hist.items()))}")

    print(f"\n=== VERDICT ===")
    for k, v in report["verdict"].items():
        print(f"  {k}: {v}")

    print(f"\n[audit] report -> {args.output_report}")


if __name__ == "__main__":
    main()
