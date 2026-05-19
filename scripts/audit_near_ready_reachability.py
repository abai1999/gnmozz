from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from evaluate_rlbench import load_pose_field_scorer
from train_near_ready_residual_adapter import NearReadyDataset, run_reachability_audit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_npz", required=True)
    ap.add_argument("--baseline_ckpt", required=True)
    ap.add_argument("--batch_size", type=int, default=64)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = NearReadyDataset(args.dataset_npz)
    baseline = load_pose_field_scorer(args.baseline_ckpt)
    baseline.eval()
    for p in baseline.parameters():
        p.requires_grad = False
    metrics = run_reachability_audit(dataset, baseline, device, args.batch_size)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
