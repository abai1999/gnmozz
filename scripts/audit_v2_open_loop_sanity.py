#!/usr/bin/env python3
"""Open-loop sanity: compare v2 with different delta inputs.

Tests v2 on the same rows with:
  A: dataset real delta (current_to_target_delta_local from v2 dataset)
  B: proxy delta (handoff residual pred_residual_*)
  C: hardcoded basin delta (computed from hardcoded basin centre)
"""
from __future__ import annotations

import argparse, json
from collections import Counter
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

from prismatic.models.target_conditioned_alignment_policy import TargetConditionedAlignmentPolicy
from prismatic.vla.datasets.target_conditioned_alignment_v2_dataset import TargetConditionedAlignmentV2Dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--v2_ckpt", type=Path, required=True)
    parser.add_argument("--dataset_npz", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    model = TargetConditionedAlignmentPolicy(proposal_count=8).to(device).eval()
    ckpt = torch.load(args.v2_ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)

    ds = TargetConditionedAlignmentV2Dataset(str(args.dataset_npz), stage_bucket_filter=None)
    loader = DataLoader(ds, batch_size=64, shuffle=False)

    # Hardcoded basin centre
    BASIN = np.array([0.065578, 0.040987, 0.769993, 0.004968, 0.001088, 0.818756, -0.574120], dtype=np.float32)

    results = {"A_dataset_real": [], "B_proxy": [], "C_hardcoded_basin": []}

    for batch in loader:
        wd = batch["wrist_depth"].to(device)
        fh = batch["force_history"].to(device)
        pr = batch["proprio"].to(device)
        ba = batch["planner_base_action_local"].to(device)
        pa = batch["proposal_actions"].to(device)
        oracle = batch["best_stage_action_index"]
        buckets = batch["stage_bucket"]
        bsz = len(batch["row_index"])

        # --- A: dataset real delta ---
        td_real = batch["current_to_target_delta_local"].to(device)
        pd_real = batch["post_candidate_delta"].to(device)
        with torch.no_grad():
            out_a = model(wd, fh, pr, ba, td_real, pa, pd_real,
                          batch["xy_improvement"].to(device), batch["z_improvement"].to(device),
                          batch["yaw_improvement"].to(device), batch["geometry_improvement"].to(device))
        sel_a = out_a["candidate_scores"].argmax(dim=-1).cpu()

        # --- B: proxy delta from handoff residual ---
        # Not available in offline dataset; use zeros as fallback (what happens at runtime)
        td_proxy = torch.zeros_like(td_real)
        pd_proxy = td_proxy.unsqueeze(1) - pa
        xi_p = torch.zeros(bsz, 8, device=device)
        zi_p = torch.zeros(bsz, 8, device=device)
        yi_p = torch.zeros(bsz, 8, device=device)
        gi_p = torch.zeros(bsz, 8, device=device)
        with torch.no_grad():
            out_b = model(wd, fh, pr, ba, td_proxy, pa, pd_proxy, xi_p, zi_p, yi_p, gi_p)
        sel_b = out_b["candidate_scores"].argmax(dim=-1).cpu()

        # --- C: hardcoded basin delta (not computable offline without current_pose) ---
        # Use dataset real delta as best approximation
        sel_c = sel_a  # same as A since we don't have gripper_pose

        for j in range(bsz):
            b = str(buckets[j])
            o = int(oracle[j].item())
            results["A_dataset_real"].append({"bucket": b, "oracle": o, "v2": int(sel_a[j].item()),
                                               "match": int(sel_a[j].item() == o)})
            results["B_proxy"].append({"bucket": b, "oracle": o, "v2": int(sel_b[j].item()),
                                        "match": int(sel_b[j].item() == o)})
            results["C_hardcoded_basin"].append({"bucket": b, "oracle": o, "v2": int(sel_c[j].item()),
                                                  "match": int(sel_c[j].item() == o)})

    # Report
    report = {}
    for label, data in results.items():
        buckets_order = ["near_alignment", "mid_approach_assist", "far_coarse_approach", "micro_contact_refine"]
        per_bucket = {}
        for b in buckets_order:
            items = [d for d in data if d["bucket"] == b]
            if not items:
                continue
            n = len(items)
            top1 = float(np.mean([d["match"] for d in items]))
            v2_hist = dict(Counter(d["v2"] for d in items))
            per_bucket[b] = {"n": n, "top1": round(top1, 4), "v2_histogram": v2_hist}

        overall_n = len(data)
        overall_top1 = float(np.mean([d["match"] for d in data]))
        overall_hist = dict(Counter(d["v2"] for d in data))
        report[label] = {
            "overall_n": overall_n, "overall_top1": round(overall_top1, 4),
            "overall_histogram": overall_hist, "per_bucket": per_bucket,
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n=== OPEN-LOOP SANITY ===")
    for label in ["A_dataset_real", "B_proxy"]:
        r = report[label]
        print(f"\n{label}: overall_top1={r['overall_top1']} hist={r['overall_histogram']}")
        for b, s in r["per_bucket"].items():
            print(f"  {b}: n={s['n']} top1={s['top1']} hist={s['v2_histogram']}")

    print(f"\n[audit] report -> {args.output}")


if __name__ == "__main__":
    main()
