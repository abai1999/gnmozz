"""
Train a band-limited near-ready residual score adapter on top of a frozen
baseline pose-field scorer. The adapter only learns local candidate score
corrections inside the frozen baseline-selected group.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import BatchSampler, DataLoader, Dataset, Subset

from prismatic.models.near_ready_residual_adapter import NearReadyResidualScoreAdapter
from evaluate_rlbench import load_pose_field_scorer


class NearReadyDataset(Dataset):
    def __init__(self, npz_path: str):
        data = np.load(npz_path, allow_pickle=False)
        self.data = {k: data[k] for k in data.files}

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


def split_by_episode(dataset: NearReadyDataset, val_ratio: float, seed: int):
    if "episode_index" not in dataset.data:
        raise RuntimeError("episode-level split is mandatory, but dataset is missing `episode_index`.")
    episode_ids = np.asarray(dataset.data["episode_index"], dtype=np.int64)
    unique_eps = np.unique(episode_ids)
    rng = np.random.default_rng(seed)
    rng.shuffle(unique_eps)
    val_count = max(1, int(round(len(unique_eps) * val_ratio)))
    val_eps = set(unique_eps[:val_count].tolist())
    train_idx = [i for i, ep in enumerate(episode_ids.tolist()) if ep not in val_eps]
    val_idx = [i for i, ep in enumerate(episode_ids.tolist()) if ep in val_eps]
    if not train_idx or not val_idx:
        mid = max(1, len(dataset) // 5)
        val_idx = list(range(mid))
        train_idx = list(range(mid, len(dataset)))
    return train_idx, val_idx


class ReadyAwareBatchSampler(BatchSampler):
    """Ensures each batch contains at least one ready-support row when available."""

    def __init__(self, indices, ready_flags, batch_size: int, shuffle: bool, seed: int):
        self.indices = np.asarray(indices, dtype=np.int64)
        self.ready_flags = np.asarray(ready_flags, dtype=np.float32)[self.indices] > 0.5
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self._epoch = 0

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self._epoch)
        self._epoch += 1
        ready_pool = self.indices[self.ready_flags].copy()
        other_pool = self.indices[~self.ready_flags].copy()
        if self.shuffle:
            rng.shuffle(ready_pool)
            rng.shuffle(other_pool)
        ready_pos = 0
        other_pos = 0
        has_ready = ready_pool.size > 0
        batches = []
        while ready_pos < ready_pool.size or other_pos < other_pool.size:
            batch = []
            if has_ready and ready_pos < ready_pool.size:
                batch.append(int(ready_pool[ready_pos]))
                ready_pos += 1
            while len(batch) < self.batch_size and other_pos < other_pool.size:
                batch.append(int(other_pool[other_pos]))
                other_pos += 1
            while len(batch) < self.batch_size and ready_pos < ready_pool.size:
                batch.append(int(ready_pool[ready_pos]))
                ready_pos += 1
            if batch:
                if self.shuffle:
                    rng.shuffle(batch)
                batches.append(batch)
        if self.shuffle:
            rng.shuffle(batches)
        for batch in batches:
            yield batch

    def __len__(self):
        return int(np.ceil(len(self.indices) / max(self.batch_size, 1)))


def _to_device(batch, device):
    return {k: v.to(device) for k, v in batch.items()}


@torch.no_grad()
def _baseline_outputs(baseline, batch, device):
    fr = batch["front_rgb"].to(device=device, dtype=torch.float32)
    wr = batch["wrist_rgb"].to(device=device, dtype=torch.float32)
    if fr.ndim != 4 or wr.ndim != 4:
        raise RuntimeError(f"expected RGB batch tensors with 4 dims, got front={tuple(fr.shape)} wrist={tuple(wr.shape)}")
    if fr.shape[1] != 3 and fr.shape[-1] == 3:
        fr = fr.permute(0, 3, 1, 2).contiguous()
    if wr.shape[1] != 3 and wr.shape[-1] == 3:
        wr = wr.permute(0, 3, 1, 2).contiguous()
    if fr.shape[1] != 3 or wr.shape[1] != 3:
        raise RuntimeError(f"baseline scorer expects BCHW RGB input, got front={tuple(fr.shape)} wrist={tuple(wr.shape)}")
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
    outputs = baseline(
        fr, wr, wd, pr, ba, gc, si, ca,
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
    return outputs


@torch.no_grad()
def run_reachability_audit(dataset: NearReadyDataset, baseline, device, batch_size: int):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate)
    reachable = 0
    total = 0
    regret_gaps = []
    teacher_better_margin_gaps = []
    group_margins = []
    weak_group_margin_count = 0
    for batch in loader:
        batch = _to_device(batch, device)
        outputs = _baseline_outputs(baseline, batch, device)
        scores = outputs["candidate_scores"]
        group_logits = outputs["group_logits"]
        cgi = batch["candidate_group_index"].long()
        cmask = batch["candidate_mask"] > 0.5
        oracle = batch["candidate_oracle_score"]
        oracle_valid = oracle > -1e8
        group_valid = []
        for gid in range(group_logits.shape[1]):
            group_valid.append(torch.any(cgi.eq(gid) & cmask, dim=1))
        group_valid = torch.stack(group_valid, dim=1)
        masked_group_logits = group_logits.masked_fill(~group_valid, -1e9)
        pred_group = masked_group_logits.argmax(dim=-1)
        top2 = torch.topk(masked_group_logits, k=min(2, masked_group_logits.shape[1]), dim=-1).values
        if top2.shape[1] == 1:
            margin = torch.full_like(top2[:, 0], 1e9)
        else:
            margin = top2[:, 0] - top2[:, 1]
        group_mask = cgi.eq(pred_group.unsqueeze(1)) & cmask & oracle_valid
        best_overall = oracle.masked_fill(~(cmask & oracle_valid), -1e9).argmax(dim=-1)
        best_reachable = oracle.masked_fill(~group_mask, -1e9).argmax(dim=-1)
        baseline_held = scores.masked_fill(~group_mask, -1e9).argmax(dim=-1)
        for i in range(scores.shape[0]):
            total += 1
            bo = int(best_overall[i].item())
            br = int(best_reachable[i].item())
            bh = int(baseline_held[i].item())
            if bool(group_mask[i, bo].item()):
                reachable += 1
            regret_gaps.append(float((oracle[i, bo] - oracle[i, br]).item()))
            teacher_better_margin_gaps.append(float((oracle[i, br] - oracle[i, bh]).item()))
            mg = float(margin[i].item())
            group_margins.append(mg)
            if mg < 0.35:
                weak_group_margin_count += 1
    return {
        "total_rows": int(total),
        "reachable_ratio": float(reachable / max(total, 1)),
        "mean_regret_gap_overall_vs_reachable": float(np.mean(regret_gaps) if regret_gaps else 0.0),
        "p95_regret_gap_overall_vs_reachable": float(np.percentile(regret_gaps, 95) if regret_gaps else 0.0),
        "mean_teacher_better_gap_vs_baseline_held": float(np.mean(teacher_better_margin_gaps) if teacher_better_margin_gaps else 0.0),
        "p95_teacher_better_gap_vs_baseline_held": float(np.percentile(teacher_better_margin_gaps, 95) if teacher_better_margin_gaps else 0.0),
        "group_top1_top2_margin_mean": float(np.mean(group_margins) if group_margins else 0.0),
        "group_top1_top2_margin_p50": float(np.percentile(group_margins, 50) if group_margins else 0.0),
        "group_top1_top2_margin_p90": float(np.percentile(group_margins, 90) if group_margins else 0.0),
        "group_top1_top2_margin_p95": float(np.percentile(group_margins, 95) if group_margins else 0.0),
        "group_top1_top2_margin_lt_0p35_ratio": float(weak_group_margin_count / max(total, 1)),
    }


def evaluate(adapter, dataset_subset, baseline, device, batch_size: int, clip_rho: float, rank_margin: float):
    loader = DataLoader(dataset_subset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate)
    total = 0
    correct = 0
    switch_count = 0
    margin_hits = 0
    xy_focus_correct = 0
    xy_focus_total = 0
    yaw_focus_correct = 0
    yaw_focus_total = 0
    regrets = []
    for batch in loader:
        batch = _to_device(batch, device)
        base = _baseline_outputs(baseline, batch, device)
        base_scores = base["candidate_scores"].detach()
        group_logits = base["group_logits"].detach()
        cgi = batch["candidate_group_index"].long()
        cmask = batch["candidate_mask"] > 0.5
        oracle = batch["candidate_oracle_score"]
        oracle_valid = oracle > -1e8

        group_valid = []
        for gid in range(group_logits.shape[1]):
            group_valid.append(torch.any(cgi.eq(gid) & cmask, dim=1))
        group_valid = torch.stack(group_valid, dim=1)
        pred_group = group_logits.masked_fill(~group_valid, -1e9).argmax(dim=-1)
        group_mask = cgi.eq(pred_group.unsqueeze(1)) & cmask & oracle_valid
        base_idx = base_scores.masked_fill(~group_mask, -1e9).argmax(dim=-1)
        teacher_idx = oracle.masked_fill(~group_mask, -1e9).argmax(dim=-1)

        residual = adapter(
            candidate_actions=batch["candidate_actions_local"].to(device=device, dtype=torch.float32),
            gripper_context=batch["gripper_context"].to(device=device, dtype=torch.float32),
            phase_age=batch["phase_age"].to(device=device, dtype=torch.float32),
            steps_since_last_replan=batch["steps_since_last_replan"].to(device=device, dtype=torch.float32),
            current_delta_basin_target=batch["proxy_current_delta_basin_target"].to(device=device, dtype=torch.float32),
            current_dx_sign=batch["current_dx_sign"].to(device=device, dtype=torch.long),
            current_dy_sign=batch["current_dy_sign"].to(device=device, dtype=torch.long),
            current_dyaw_sign=batch["current_dyaw_sign"].to(device=device, dtype=torch.long),
            basin_distance_bin=batch["basin_distance_bin"].to(device=device, dtype=torch.long),
            candidate_mask=batch["candidate_mask"].to(device=device, dtype=torch.float32),
        )
        final_scores = base_scores + torch.clamp(residual, min=-clip_rho, max=clip_rho)
        pred_idx = final_scores.masked_fill(~group_mask, -1e9).argmax(dim=-1)
        teacher_score = final_scores.gather(1, teacher_idx.unsqueeze(1)).squeeze(1)
        base_score = final_scores.gather(1, base_idx.unsqueeze(1)).squeeze(1)
        for i in range(final_scores.shape[0]):
            total += 1
            pi = int(pred_idx[i].item())
            ti = int(teacher_idx[i].item())
            bi = int(base_idx[i].item())
            if pi == ti:
                correct += 1
            if pi != bi:
                switch_count += 1
            if float((teacher_score[i] - base_score[i]).item()) >= rank_margin:
                margin_hits += 1
            regrets.append(float((oracle[i, ti] - oracle[i, pi]).item()))
            axis = int(batch["dominant_axis_bucket"][i].item()) if "dominant_axis_bucket" in batch else 0
            if axis in (1, 3):
                xy_focus_total += 1
                if pi == ti:
                    xy_focus_correct += 1
            if axis in (2, 3):
                yaw_focus_total += 1
                if pi == ti:
                    yaw_focus_correct += 1
    return {
        "rows": int(total),
        "top1_reachable_acc": float(correct / max(total, 1)),
        "candidate_switch_rate": float(switch_count / max(total, 1)),
        "teacher_better_margin_hit_rate": float(margin_hits / max(total, 1)),
        "oracle_regret_mean": float(np.mean(regrets) if regrets else 0.0),
        "oracle_regret_p95": float(np.percentile(regrets, 95) if regrets else 0.0),
        "xy_focus_acc": float(xy_focus_correct / max(xy_focus_total, 1)),
        "yaw_focus_acc": float(yaw_focus_correct / max(yaw_focus_total, 1)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_npz", required=True)
    ap.add_argument("--baseline_ckpt", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--val_ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--clip_rho", type=float, default=0.35)
    ap.add_argument("--rank_margin", type=float, default=0.35)
    ap.add_argument("--lambda_ce", type=float, default=1.0)
    ap.add_argument("--lambda_pair", type=float, default=1.0)
    ap.add_argument("--lambda_margin", type=float, default=1.0)
    ap.add_argument("--lambda_l2", type=float, default=0.02)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = NearReadyDataset(args.dataset_npz)
    train_idx, val_idx = split_by_episode(dataset, args.val_ratio, args.seed)
    val_ds = Subset(dataset, val_idx)
    if "ready_support" not in dataset.data:
        raise RuntimeError("near-ready dataset is missing `ready_support`; batch-level ready visibility cannot be enforced.")
    train_batch_sampler = ReadyAwareBatchSampler(
        train_idx,
        dataset.data["ready_support"],
        batch_size=args.batch_size,
        shuffle=True,
        seed=args.seed,
    )
    train_loader = DataLoader(dataset, batch_sampler=train_batch_sampler, num_workers=0, collate_fn=collate)

    baseline = load_pose_field_scorer(args.baseline_ckpt)
    baseline.eval()
    for p in baseline.parameters():
        p.requires_grad = False

    audit = run_reachability_audit(dataset, baseline, device, args.batch_size)
    print("[residual_adapter] reachability_audit")
    print(json.dumps(audit, indent=2))

    model = NearReadyResidualScoreAdapter(clip_rho=float(args.clip_rho)).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    history = []
    best_metric = -1e9
    best_state = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_rows = 0
        for batch in train_loader:
            batch = _to_device(batch, device)
            with torch.no_grad():
                base = _baseline_outputs(baseline, batch, device)
                base_scores = base["candidate_scores"].detach()
                group_logits = base["group_logits"].detach()
            cgi = batch["candidate_group_index"].long()
            cmask = batch["candidate_mask"] > 0.5
            oracle = batch["candidate_oracle_score"]
            oracle_valid = oracle > -1e8

            group_valid = []
            for gid in range(group_logits.shape[1]):
                group_valid.append(torch.any(cgi.eq(gid) & cmask, dim=1))
            group_valid = torch.stack(group_valid, dim=1)
            pred_group = group_logits.masked_fill(~group_valid, -1e9).argmax(dim=-1)
            group_mask = cgi.eq(pred_group.unsqueeze(1)) & cmask & oracle_valid

            residual = model(
                candidate_actions=batch["candidate_actions_local"].to(device=device, dtype=torch.float32),
                gripper_context=batch["gripper_context"].to(device=device, dtype=torch.float32),
                phase_age=batch["phase_age"].to(device=device, dtype=torch.float32),
                steps_since_last_replan=batch["steps_since_last_replan"].to(device=device, dtype=torch.float32),
                current_delta_basin_target=batch["proxy_current_delta_basin_target"].to(device=device, dtype=torch.float32),
                current_dx_sign=batch["current_dx_sign"].to(device=device, dtype=torch.long),
                current_dy_sign=batch["current_dy_sign"].to(device=device, dtype=torch.long),
                current_dyaw_sign=batch["current_dyaw_sign"].to(device=device, dtype=torch.long),
                basin_distance_bin=batch["basin_distance_bin"].to(device=device, dtype=torch.long),
                candidate_mask=batch["candidate_mask"].to(device=device, dtype=torch.float32),
            )
            final_scores = base_scores + torch.clamp(residual, min=-args.clip_rho, max=args.clip_rho)
            teacher_idx = oracle.masked_fill(~group_mask, -1e9).argmax(dim=-1)
            baseline_idx = base_scores.masked_fill(~group_mask, -1e9).argmax(dim=-1)

            masked_final = final_scores.masked_fill(~group_mask, -1e9)
            loss_ce = F.cross_entropy(masked_final, teacher_idx, reduction="none")

            teacher_score = final_scores.gather(1, teacher_idx.unsqueeze(1)).squeeze(1)
            baseline_score = final_scores.gather(1, baseline_idx.unsqueeze(1)).squeeze(1)
            oracle_teacher = oracle.gather(1, teacher_idx.unsqueeze(1)).squeeze(1)
            oracle_baseline = oracle.gather(1, baseline_idx.unsqueeze(1)).squeeze(1)
            teacher_better = oracle_teacher > oracle_baseline + 1e-6
            loss_margin = F.relu(float(args.rank_margin) - (teacher_score - baseline_score))
            loss_margin = loss_margin * teacher_better.float()

            pair_mask = group_mask & (oracle < oracle_teacher.unsqueeze(1) - 1e-6)
            pair_gap = (oracle_teacher.unsqueeze(1) - oracle).clamp_min(0.0)
            pair_penalty = F.relu(float(args.rank_margin) - (teacher_score.unsqueeze(1) - final_scores))
            pair_num = (pair_penalty * pair_gap * pair_mask.float()).sum(dim=1)
            pair_den = torch.clamp((pair_gap * pair_mask.float()).sum(dim=1), min=1e-6)
            loss_pair = pair_num / pair_den

            sample_weight = batch["sample_weight"].to(device=device, dtype=torch.float32)
            loss_l2 = residual.pow(2).mean(dim=1)
            loss = (
                (loss_ce * sample_weight).sum() / torch.clamp(sample_weight.sum(), min=1e-6) * float(args.lambda_ce)
                + (loss_pair * sample_weight).sum() / torch.clamp(sample_weight.sum(), min=1e-6) * float(args.lambda_pair)
                + (loss_margin * sample_weight).sum() / torch.clamp(sample_weight.sum(), min=1e-6) * float(args.lambda_margin)
                + (loss_l2 * sample_weight).sum() / torch.clamp(sample_weight.sum(), min=1e-6) * float(args.lambda_l2)
            )
            optim.zero_grad()
            loss.backward()
            optim.step()
            total_loss += float(loss.item()) * int(sample_weight.shape[0])
            total_rows += int(sample_weight.shape[0])

        val_metrics = evaluate(model, val_ds, baseline, device, args.batch_size, args.clip_rho, args.rank_margin)
        row = {
            "epoch": int(epoch),
            "train_loss": float(total_loss / max(total_rows, 1)),
            **val_metrics,
        }
        history.append(row)
        metric = val_metrics["top1_reachable_acc"] - 0.25 * val_metrics["oracle_regret_mean"]
        print(f"[residual_adapter] epoch={epoch} loss={row['train_loss']:.4f} val_top1={val_metrics['top1_reachable_acc']:.4f} val_regret={val_metrics['oracle_regret_mean']:.4f} val_switch={val_metrics['candidate_switch_rate']:.4f} val_margin_hit={val_metrics['teacher_better_margin_hit_rate']:.4f}")
        if metric > best_metric:
            best_metric = metric
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "module_type": "near_ready_residual_adapter",
        "model_state_dict": best_state if best_state is not None else {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "dataset_npz": args.dataset_npz,
        "baseline_ckpt": args.baseline_ckpt,
        "clip_rho": float(args.clip_rho),
        "rank_margin": float(args.rank_margin),
        "reachability_audit": audit,
    }
    torch.save(ckpt, output_dir / "near_ready_residual_adapter_best.pt")
    torch.save({**ckpt, "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()}}, output_dir / "near_ready_residual_adapter_final.pt")
    (output_dir / "history.json").write_text(json.dumps(history, indent=2))
    (output_dir / "reachability_audit.json").write_text(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
