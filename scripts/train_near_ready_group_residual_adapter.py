"""
Train near-ready gated group-logit residual adapter on top of a frozen baseline
pose-field scorer.

Primary goal: fix wrong baseline group selection in near-ready band without
changing planner / step-scale / ready timing contracts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset

from evaluate_rlbench import load_pose_field_scorer
from prismatic.models.near_ready_residual_adapter import NearReadyGroupResidualAdapter


class NpzDataset(Dataset):
    def __init__(self, npz_path: str):
        d = np.load(npz_path, allow_pickle=False)
        self.data = {k: d[k] for k in d.files}
        if "episode_index" not in self.data:
            raise RuntimeError("dataset must contain `episode_index` for episode-level split.")

    def __len__(self):
        return int(self.data["candidate_actions_local"].shape[0])

    def __getitem__(self, idx):
        out = {}
        for k, v in self.data.items():
            out[k] = torch.from_numpy(np.asarray(v[idx]))
        return out


def collate(batch):
    out = {}
    for key in batch[0].keys():
        out[key] = torch.stack([row[key] for row in batch], dim=0)
    return out


def split_by_episode(dataset: NpzDataset, val_ratio: float, seed: int):
    ep = np.asarray(dataset.data["episode_index"], dtype=np.int64)
    uniq = np.unique(ep)
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    val_n = max(1, int(round(len(uniq) * val_ratio)))
    val_set = set(uniq[:val_n].tolist())
    train_idx = [i for i, e in enumerate(ep.tolist()) if e not in val_set]
    val_idx = [i for i, e in enumerate(ep.tolist()) if e in val_set]
    return train_idx, val_idx


def _to_device(batch, device):
    return {k: v.to(device) for k, v in batch.items()}


@torch.no_grad()
def _baseline_outputs(baseline, batch, device):
    fr = batch["front_rgb"].to(device=device, dtype=torch.float32)
    wr = batch["wrist_rgb"].to(device=device, dtype=torch.float32)
    if fr.ndim == 4 and fr.shape[1] != 3 and fr.shape[-1] == 3:
        fr = fr.permute(0, 3, 1, 2).contiguous()
    if wr.ndim == 4 and wr.shape[1] != 3 and wr.shape[-1] == 3:
        wr = wr.permute(0, 3, 1, 2).contiguous()
    if fr.max() > 1.5:
        fr = fr / 255.0
    if wr.max() > 1.5:
        wr = wr / 255.0
    wd = batch["wrist_depth"].to(device=device, dtype=torch.float32)
    pr = batch["proprio"].to(device=device, dtype=torch.float32)
    ba = batch["base_action"].to(device=device, dtype=torch.float32)
    gc = batch["gripper_context"].to(device=device, dtype=torch.float32)
    si = batch["step_idx"].to(device=device, dtype=torch.long)
    pid = batch["phase_id"].to(device=device, dtype=torch.long)
    page = batch["phase_age"].to(device=device, dtype=torch.float32)
    sr = batch["steps_since_last_replan"].to(device=device, dtype=torch.float32)
    ca = batch["candidate_actions_local"].to(device=device, dtype=torch.float32)
    cmask = batch["candidate_mask"].to(device=device, dtype=torch.float32)
    cur_delta = batch["proxy_current_delta_basin_target"].to(device=device, dtype=torch.float32)
    dxs = batch["current_dx_sign"].to(device=device, dtype=torch.long)
    dys = batch["current_dy_sign"].to(device=device, dtype=torch.long)
    dyaws = batch["current_dyaw_sign"].to(device=device, dtype=torch.long)
    basin_bin = batch["basin_distance_bin"].to(device=device, dtype=torch.long)
    return baseline(
        fr,
        wr,
        wd,
        pr,
        ba,
        gc,
        si,
        ca,
        phase_id=pid,
        phase_age=page,
        steps_since_last_replan=sr,
        current_delta_basin_target=cur_delta,
        current_dx_sign=dxs,
        current_dy_sign=dys,
        current_dyaw_sign=dyaws,
        basin_distance_bin=basin_bin,
        candidate_mask=cmask,
        return_aux=True,
    )


def _group_valid_mask(candidate_group_index: torch.Tensor, candidate_mask: torch.Tensor, num_groups: int):
    valid = candidate_mask > 0.5
    out = []
    for gid in range(num_groups):
        out.append(torch.any(candidate_group_index.eq(gid) & valid, dim=1))
    return torch.stack(out, dim=1)


def _teacher_group_targets(oracle_scores: torch.Tensor, candidate_group_index: torch.Tensor, candidate_mask: torch.Tensor, num_groups: int):
    valid = candidate_mask > 0.5
    oracle_valid = oracle_scores > -1e8
    teacher_group = []
    group_best = torch.full((oracle_scores.shape[0], num_groups), -1e9, device=oracle_scores.device, dtype=oracle_scores.dtype)
    for gid in range(num_groups):
        gm = candidate_group_index.eq(gid) & valid & oracle_valid
        group_best[:, gid] = oracle_scores.masked_fill(~gm, -1e9).max(dim=1).values
    teacher_group = group_best.argmax(dim=1)
    return teacher_group, group_best


@torch.no_grad()
def evaluate(model, dataset_subset, baseline, device, batch_size: int, clip_rho_g: float):
    loader = DataLoader(dataset_subset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate)
    total = 0
    group_flip = 0
    flip_to_teacher = 0
    base_teacher_disagree = 0
    reachable = 0
    regret_gaps = []
    for batch in loader:
        batch = _to_device(batch, device)
        base = _baseline_outputs(baseline, batch, device)
        base_group_logits = base["group_logits"].detach()
        cgi = batch["candidate_group_index"].long()
        cmask = batch["candidate_mask"] > 0.5
        oracle = batch["candidate_oracle_score"]
        num_groups = int(base_group_logits.shape[1])

        gvalid = _group_valid_mask(cgi, cmask, num_groups)
        base_group = base_group_logits.masked_fill(~gvalid, -1e9).argmax(dim=-1)
        teacher_group, group_best = _teacher_group_targets(oracle, cgi, cmask, num_groups)

        gres = model(
            gripper_context=batch["gripper_context"].to(device=device, dtype=torch.float32),
            phase_age=batch["phase_age"].to(device=device, dtype=torch.float32),
            steps_since_last_replan=batch["steps_since_last_replan"].to(device=device, dtype=torch.float32),
            current_delta_basin_target=batch["proxy_current_delta_basin_target"].to(device=device, dtype=torch.float32),
            current_dx_sign=batch["current_dx_sign"].to(device=device, dtype=torch.long),
            current_dy_sign=batch["current_dy_sign"].to(device=device, dtype=torch.long),
            current_dyaw_sign=batch["current_dyaw_sign"].to(device=device, dtype=torch.long),
            basin_distance_bin=batch["basin_distance_bin"].to(device=device, dtype=torch.long),
            group_valid_mask=gvalid,
        )
        final_group_logits = base_group_logits + torch.clamp(gres, min=-clip_rho_g, max=clip_rho_g)
        pred_group = final_group_logits.masked_fill(~gvalid, -1e9).argmax(dim=-1)

        overall_best = oracle.masked_fill(~cmask, -1e9).max(dim=1).values
        reachable_best = group_best.gather(1, pred_group.unsqueeze(1)).squeeze(1)
        base_best = group_best.gather(1, base_group.unsqueeze(1)).squeeze(1)

        for i in range(oracle.shape[0]):
            total += 1
            bg = int(base_group[i].item())
            pg = int(pred_group[i].item())
            tg = int(teacher_group[i].item())
            if bg != pg:
                group_flip += 1
            if bg != tg:
                base_teacher_disagree += 1
            if bg != tg and pg == tg:
                flip_to_teacher += 1
            if pg == tg:
                reachable += 1
            regret_gaps.append(float((overall_best[i] - reachable_best[i]).item()))

    return {
        "rows": int(total),
        "group_switch_count": int(group_flip),
        "group_switch_rate": float(group_flip / max(total, 1)),
        "baseline_teacher_disagree_count": int(base_teacher_disagree),
        "wrong_baseline_to_teacher_flip_count": int(flip_to_teacher),
        "wrong_baseline_to_teacher_flip_rate": float(flip_to_teacher / max(base_teacher_disagree, 1)),
        "reachable_ratio": float(reachable / max(total, 1)),
        "mean_regret_gap_overall_vs_reachable": float(np.mean(regret_gaps) if regret_gaps else 0.0),
        "p95_regret_gap_overall_vs_reachable": float(np.percentile(regret_gaps, 95) if regret_gaps else 0.0),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_npz", required=True)
    ap.add_argument("--baseline_ckpt", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--val_ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--clip_rho_g", type=float, default=0.45)
    ap.add_argument("--rank_margin_g", type=float, default=0.35)
    ap.add_argument("--lambda_ce", type=float, default=1.0)
    ap.add_argument("--lambda_margin", type=float, default=1.0)
    ap.add_argument("--lambda_l2", type=float, default=0.02)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = NpzDataset(args.dataset_npz)
    train_idx, val_idx = split_by_episode(dataset, args.val_ratio, args.seed)
    train_ds = Subset(dataset, train_idx)
    val_ds = Subset(dataset, val_idx)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=collate)

    baseline = load_pose_field_scorer(args.baseline_ckpt)
    baseline.eval()
    for p in baseline.parameters():
        p.requires_grad = False

    num_groups = int(getattr(baseline, "num_candidate_groups", 37))
    model = NearReadyGroupResidualAdapter(num_groups=num_groups, clip_rho=float(args.clip_rho_g)).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    history = []
    best_metric = -1e9
    best_state = None
    best_eval = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        row_count = 0
        for batch in train_loader:
            batch = _to_device(batch, device)
            with torch.no_grad():
                base = _baseline_outputs(baseline, batch, device)
                base_group_logits = base["group_logits"].detach()
            cgi = batch["candidate_group_index"].long()
            cmask = batch["candidate_mask"] > 0.5
            oracle = batch["candidate_oracle_score"]
            gvalid = _group_valid_mask(cgi, cmask, num_groups)
            base_group = base_group_logits.masked_fill(~gvalid, -1e9).argmax(dim=-1)
            teacher_group, group_best = _teacher_group_targets(oracle, cgi, cmask, num_groups)

            gres = model(
                gripper_context=batch["gripper_context"].to(device=device, dtype=torch.float32),
                phase_age=batch["phase_age"].to(device=device, dtype=torch.float32),
                steps_since_last_replan=batch["steps_since_last_replan"].to(device=device, dtype=torch.float32),
                current_delta_basin_target=batch["proxy_current_delta_basin_target"].to(device=device, dtype=torch.float32),
                current_dx_sign=batch["current_dx_sign"].to(device=device, dtype=torch.long),
                current_dy_sign=batch["current_dy_sign"].to(device=device, dtype=torch.long),
                current_dyaw_sign=batch["current_dyaw_sign"].to(device=device, dtype=torch.long),
                basin_distance_bin=batch["basin_distance_bin"].to(device=device, dtype=torch.long),
                group_valid_mask=gvalid,
            )
            final_group_logits = base_group_logits + torch.clamp(gres, min=-args.clip_rho_g, max=args.clip_rho_g)
            masked_final = final_group_logits.masked_fill(~gvalid, -1e9)
            loss_ce = F.cross_entropy(masked_final, teacher_group, reduction="none")

            teacher_score = masked_final.gather(1, teacher_group.unsqueeze(1)).squeeze(1)
            baseline_score = masked_final.gather(1, base_group.unsqueeze(1)).squeeze(1)
            teacher_best = group_best.gather(1, teacher_group.unsqueeze(1)).squeeze(1)
            baseline_best = group_best.gather(1, base_group.unsqueeze(1)).squeeze(1)
            teacher_better = teacher_best > baseline_best + 1e-6
            loss_margin = F.relu(float(args.rank_margin_g) - (teacher_score - baseline_score)) * teacher_better.float()

            sw = batch["sample_weight"].to(device=device, dtype=torch.float32) if "sample_weight" in batch else torch.ones_like(loss_ce)
            loss_l2 = gres.pow(2).mean(dim=1)
            denom = torch.clamp(sw.sum(), min=1e-6)
            loss = (
                (loss_ce * sw).sum() / denom * float(args.lambda_ce)
                + (loss_margin * sw).sum() / denom * float(args.lambda_margin)
                + (loss_l2 * sw).sum() / denom * float(args.lambda_l2)
            )
            optim.zero_grad()
            loss.backward()
            optim.step()
            loss_sum += float(loss.item()) * int(sw.shape[0])
            row_count += int(sw.shape[0])

        val_metrics = evaluate(model, val_ds, baseline, device, args.batch_size, args.clip_rho_g)
        row = {
            "epoch": int(epoch),
            "train_loss": float(loss_sum / max(row_count, 1)),
            **val_metrics,
        }
        history.append(row)
        metric = val_metrics["reachable_ratio"] + 0.5 * val_metrics["wrong_baseline_to_teacher_flip_rate"] - 0.1 * val_metrics["mean_regret_gap_overall_vs_reachable"]
        print(
            f"[group_residual] epoch={epoch} loss={row['train_loss']:.4f} "
            f"reach={val_metrics['reachable_ratio']:.4f} "
            f"flip_to_teacher={val_metrics['wrong_baseline_to_teacher_flip_rate']:.4f} "
            f"regret={val_metrics['mean_regret_gap_overall_vs_reachable']:.4f}"
        )
        if metric > best_metric:
            best_metric = metric
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            best_eval = dict(val_metrics)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "module_type": "near_ready_group_residual_adapter",
        "model_state_dict": best_state if best_state is not None else {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "dataset_npz": args.dataset_npz,
        "baseline_ckpt": args.baseline_ckpt,
        "num_groups": int(num_groups),
        "clip_rho_g": float(args.clip_rho_g),
        "rank_margin_g": float(args.rank_margin_g),
        "best_eval": best_eval if best_eval is not None else {},
    }
    torch.save(ckpt, output_dir / "near_ready_group_residual_adapter_best.pt")
    torch.save({**ckpt, "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()}}, output_dir / "near_ready_group_residual_adapter_final.pt")
    (output_dir / "history.json").write_text(json.dumps(history, indent=2))
    if best_eval is not None:
        (output_dir / "best_eval.json").write_text(json.dumps(best_eval, indent=2))


if __name__ == "__main__":
    main()

