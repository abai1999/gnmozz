from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset

from prismatic.models.pose_field_scorer import PoseFieldScorer
from prismatic.models.student_group_selector_v2 import StudentGroupSelectorV2
from prismatic.models.student_handoff_state_head_v2 import StudentHandoffStateHeadV2


class NearReadyGroupDataset(Dataset):
    def __init__(self, npz_path: str):
        arr = np.load(npz_path, allow_pickle=False)
        self.data = {k: np.asarray(arr[k]) for k in arr.files}
        self.length = int(self.data["best_group_index"].shape[0])
        if "episode_index" not in self.data:
            raise RuntimeError("dataset must contain episode_index")

    def __len__(self):
        return self.length

    def _substage_value(self, idx: int) -> int:
        if "substage_id" in self.data:
            return int(self.data["substage_id"][idx])
        return int(self.data.get("phase_id", np.ones((self.length,), dtype=np.int64))[idx])

    def __getitem__(self, idx):
        front = torch.from_numpy(self.data["front_rgb"][idx].astype(np.float32).transpose(2, 0, 1)) / 255.0
        wrist = torch.from_numpy(self.data["wrist_rgb"][idx].astype(np.float32).transpose(2, 0, 1)) / 255.0
        depth = torch.from_numpy(self.data["wrist_depth"][idx].astype(np.float32))
        if depth.ndim == 2:
            depth = depth.unsqueeze(0)
        return {
            "front_rgb": front,
            "wrist_rgb": wrist,
            "wrist_depth": depth,
            "proprio": torch.from_numpy(self.data["proprio"][idx].astype(np.float32)),
            "base_action": torch.from_numpy(self.data["base_action"][idx].astype(np.float32)),
            "gripper_context": torch.from_numpy(self.data["gripper_context"][idx].astype(np.float32)),
            "step_idx": torch.tensor(int(self.data.get("step_idx", np.zeros((self.length,), dtype=np.int64))[idx]), dtype=torch.long),
            "phase_id": torch.tensor(int(self.data.get("phase_id", np.ones((self.length,), dtype=np.int64))[idx]), dtype=torch.long),
            "phase_age": torch.tensor(float(self.data.get("phase_age", np.zeros((self.length,), dtype=np.float32))[idx]), dtype=torch.float32),
            "steps_since_last_replan": torch.tensor(
                float(self.data.get("steps_since_last_replan", np.zeros((self.length,), dtype=np.float32))[idx]),
                dtype=torch.float32,
            ),
            "proxy_current_delta_basin_target": torch.from_numpy(
                self.data.get("proxy_current_delta_basin_target", self.data["current_delta_basin_target"])[idx].astype(np.float32)
            ),
            "current_dx_sign": torch.tensor(int(self.data["current_dx_sign"][idx]), dtype=torch.long),
            "current_dy_sign": torch.tensor(int(self.data["current_dy_sign"][idx]), dtype=torch.long),
            "current_dyaw_sign": torch.tensor(int(self.data["current_dyaw_sign"][idx]), dtype=torch.long),
            "basin_distance_bin": torch.tensor(int(self.data["basin_distance_bin"][idx]), dtype=torch.long),
            "substage_id": torch.tensor(self._substage_value(idx), dtype=torch.long),
            "contact_state": torch.tensor(int(self.data.get("contact_state", np.zeros((self.length,), dtype=np.int64))[idx]), dtype=torch.long),
            "stage_target_mode": torch.tensor(int(self.data.get("stage_target_mode", np.zeros((self.length,), dtype=np.int64))[idx]), dtype=torch.long),
            "candidate_group_index": torch.from_numpy(self.data["candidate_group_index"][idx].astype(np.int64)),
            "candidate_mask": torch.from_numpy(self.data["candidate_mask"][idx].astype(np.float32)),
            "candidate_actions_local": torch.from_numpy(self.data["candidate_actions_local"][idx].astype(np.float32)),
            "candidate_oracle_score": torch.from_numpy(self.data["candidate_oracle_score"][idx].astype(np.float32)),
            "best_group_index": torch.tensor(int(self.data["best_group_index"][idx]), dtype=torch.long),
            "sample_weight": torch.tensor(float(self.data.get("sample_weight", np.ones((self.length,), dtype=np.float32))[idx]), dtype=torch.float32),
            "near_xy_hard": torch.tensor(float(self.data.get("near_xy_hard", np.zeros((self.length,), dtype=np.float32))[idx]), dtype=torch.float32),
            "near_yaw_hard": torch.tensor(float(self.data.get("near_yaw_hard", np.zeros((self.length,), dtype=np.float32))[idx]), dtype=torch.float32),
            "near_coupled": torch.tensor(float(self.data.get("near_coupled", np.zeros((self.length,), dtype=np.float32))[idx]), dtype=torch.float32),
            "ready_support": torch.tensor(float(self.data.get("ready_support", np.zeros((self.length,), dtype=np.float32))[idx]), dtype=torch.float32),
            "teacher_ready_v1": torch.tensor(float(self.data.get("teacher_ready_v1", np.zeros((self.length,), dtype=np.float32))[idx]), dtype=torch.float32),
            "xy_block_v1": torch.tensor(float(self.data.get("xy_block_v1", np.zeros((self.length,), dtype=np.float32))[idx]), dtype=torch.float32),
            "yaw_needed_v1": torch.tensor(float(self.data.get("yaw_needed_v1", np.zeros((self.length,), dtype=np.float32))[idx]), dtype=torch.float32),
            "far_negative_v1": torch.tensor(float(self.data.get("far_negative_v1", np.zeros((self.length,), dtype=np.float32))[idx]), dtype=torch.float32),
            "teacher_truth_handoff_ready": torch.tensor(
                float(self.data.get("teacher_truth_handoff_ready", np.zeros((self.length,), dtype=np.float32))[idx]),
                dtype=torch.float32,
            ),
            "episode_index": torch.tensor(int(self.data["episode_index"][idx]), dtype=torch.long),
        }


def split_by_episode(dataset: NearReadyGroupDataset, val_ratio: float, seed: int):
    ep = np.asarray(dataset.data["episode_index"], dtype=np.int64)
    uniq = np.unique(ep)
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    val_n = max(1, int(round(uniq.size * val_ratio)))
    if val_n >= uniq.size:
        val_n = max(1, uniq.size - 1)
    val_set = set(uniq[:val_n].tolist())
    train_idx = [i for i, e in enumerate(ep.tolist()) if e not in val_set]
    val_idx = [i for i, e in enumerate(ep.tolist()) if e in val_set]
    return train_idx, val_idx


def split_by_episode_balanced(
    dataset: NearReadyGroupDataset,
    val_ratio: float,
    seed: int,
    min_bucket_count: int = 25,
):
    """Episode split that tries to keep rare diagnostic buckets in validation.

    This is still a strict episode-level split.  The only difference from the
    random split is the validation episode choice: we greedily select episodes
    that cover teacher_ready / xy_block / yaw_needed / far_negative buckets so
    rare buckets do not disappear from val by accident.
    """
    data = dataset.data
    ep = np.asarray(data["episode_index"], dtype=np.int64)
    uniq = np.unique(ep)
    rng = np.random.default_rng(seed)
    val_n = max(1, int(round(uniq.size * val_ratio)))
    if val_n >= uniq.size:
        val_n = max(1, uniq.size - 1)

    bucket_keys = ("teacher_ready_v1", "xy_block_v1", "yaw_needed_v1", "far_negative_v1")
    ep_stats: dict[int, dict[str, int]] = {}
    for e in uniq.tolist():
        mask = ep == e
        ep_stats[int(e)] = {}
        for key in bucket_keys:
            vals = np.asarray(data.get(key, np.zeros((dataset.length,), dtype=np.float32)))
            ep_stats[int(e)][key] = int(np.sum(vals[mask] > 0.5))
        ep_stats[int(e)]["rows"] = int(np.sum(mask))

    total_counts = {key: sum(s[key] for s in ep_stats.values()) for key in bucket_keys}
    target_counts = {
        key: min(int(min_bucket_count), max(1, int(np.ceil(total_counts[key] * val_ratio))))
        for key in bucket_keys
        if total_counts[key] > 0
    }
    selected: list[int] = []
    current = {key: 0 for key in bucket_keys}
    remaining = uniq.tolist()
    rng.shuffle(remaining)

    def score_episode(e: int) -> tuple[float, float]:
        score = 0.0
        for key, target in target_counts.items():
            deficit = max(0, target - current[key])
            if deficit <= 0:
                continue
            score += min(deficit, ep_stats[int(e)][key]) / max(target, 1)
        # Small tie-breaker prefers denser rare-bucket episodes without making
        # this a row-count split.
        density = sum(ep_stats[int(e)][key] for key in bucket_keys) / max(ep_stats[int(e)]["rows"], 1)
        return score, density

    while len(selected) < val_n and remaining:
        best_ep = max(remaining, key=score_episode)
        selected.append(int(best_ep))
        remaining.remove(best_ep)
        for key in bucket_keys:
            current[key] += ep_stats[int(best_ep)][key]

    # Rare buckets are often episode-sparse.  A diagnostic split is only useful
    # if it does not put every positive episode for a bucket into validation.
    # When possible, move one positive episode back to train and replace it with
    # a non-positive episode.  Buckets with only one positive episode remain
    # low-confidence by construction.
    for key in bucket_keys:
        positive_eps = [int(e) for e in uniq.tolist() if ep_stats[int(e)][key] > 0]
        if len(positive_eps) <= 1:
            continue
        if not all(e in selected for e in positive_eps):
            continue
        # Prefer returning the largest positive episode to train so training is
        # not starved of the rare bucket.
        move_back = max(positive_eps, key=lambda e: ep_stats[e][key])
        replacement = next((int(e) for e in remaining if ep_stats[int(e)][key] == 0), None)
        if replacement is None:
            continue
        selected.remove(move_back)
        selected.append(replacement)
        remaining.remove(replacement)
        remaining.append(move_back)

    val_set = set(selected)
    train_idx = [i for i, e in enumerate(ep.tolist()) if e not in val_set]
    val_idx = [i for i, e in enumerate(ep.tolist()) if e in val_set]
    return train_idx, val_idx


def split_by_explicit_val_episodes(dataset: NearReadyGroupDataset, val_episodes_csv: str):
    ep = np.asarray(dataset.data["episode_index"], dtype=np.int64)
    val_set = {int(x.strip()) for x in val_episodes_csv.split(",") if x.strip()}
    if not val_set:
        raise ValueError("--val_episodes was provided but no episodes were parsed")
    present = set(np.unique(ep).astype(int).tolist())
    missing = sorted(val_set - present)
    if missing:
        raise ValueError(f"--val_episodes contains episodes not present in dataset: {missing}")
    train_idx = [i for i, e in enumerate(ep.tolist()) if e not in val_set]
    val_idx = [i for i, e in enumerate(ep.tolist()) if e in val_set]
    if not train_idx or not val_idx:
        raise ValueError("explicit episode split produced an empty train or val set")
    return train_idx, val_idx


def split_bucket_summary(dataset: NearReadyGroupDataset, indices):
    data = dataset.data
    idx = np.asarray(indices, dtype=np.int64)
    episodes = np.asarray(data["episode_index"], dtype=np.int64)
    summary = {
        "rows": int(idx.size),
        "episodes": int(np.unique(episodes[idx]).size) if idx.size else 0,
        "episode_list": [int(x) for x in sorted(np.unique(episodes[idx]).tolist())] if idx.size else [],
    }
    for key in (
        "near_xy_hard",
        "near_yaw_hard",
        "near_coupled",
        "ready_support",
        "teacher_truth_handoff_ready",
        "teacher_ready_v1",
        "xy_block_v1",
        "yaw_needed_v1",
        "far_negative_v1",
    ):
        vals = np.asarray(data.get(key, np.zeros((dataset.length,), dtype=np.float32)))
        count = int(np.sum(vals[idx] > 0.5)) if idx.size else 0
        summary[f"{key}_count"] = count
        summary[f"{key}_low_confidence"] = bool(count < 25)
    return summary


def load_handoff_model(ckpt_path: str, device):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model = StudentHandoffStateHeadV2().to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def load_pose_field_model(ckpt_path: str, device):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    num_candidate_groups = int(ckpt.get("num_candidate_groups", 11))
    model = PoseFieldScorer(
        use_depth=ckpt.get("use_depth", True),
        use_base_action=ckpt.get("use_base_action", True),
        use_proprio=ckpt.get("use_proprio", True),
        use_target_context=ckpt.get("use_target_context", True),
        use_front_rgb=ckpt.get("use_front_rgb", False),
        use_wrist_rgb=ckpt.get("use_wrist_rgb", False),
        num_candidate_groups=num_candidate_groups,
        fire_only_head=ckpt.get("fire_only_head", True),
    ).to(device)
    missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    if missing or unexpected:
        print(f"[baseline_scorer] compatibility: missing={missing}, unexpected={unexpected}")
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def extract_handoff_latent(handoff_model, batch, device):
    with torch.no_grad():
        out = handoff_model(
            front_rgb=batch["front_rgb"].to(device=device, dtype=torch.float32),
            wrist_rgb=batch["wrist_rgb"].to(device=device, dtype=torch.float32),
            wrist_depth=batch["wrist_depth"].to(device=device, dtype=torch.float32),
            proprio=batch["proprio"].to(device=device, dtype=torch.float32),
            gripper_context=batch["gripper_context"].to(device=device, dtype=torch.float32),
            proxy_current_delta_basin_target=batch["proxy_current_delta_basin_target"].to(device=device, dtype=torch.float32),
            current_dx_sign=batch["current_dx_sign"].to(device=device),
            current_dy_sign=batch["current_dy_sign"].to(device=device),
            current_dyaw_sign=batch["current_dyaw_sign"].to(device=device),
            basin_distance_bin=batch["basin_distance_bin"].to(device=device),
            substage_id=batch["substage_id"].to(device=device),
            contact_state=batch["contact_state"].to(device=device),
            stage_target_mode=batch["stage_target_mode"].to(device=device),
        )
    return out["latent"].detach()


def compute_group_valid(batch, device, num_groups: int):
    cgi = batch["candidate_group_index"].to(device=device)
    cmask = batch["candidate_mask"].to(device=device) > 0.5
    oracle_score = batch["candidate_oracle_score"].to(device=device, dtype=torch.float32)
    valid_candidate = cmask & (oracle_score > -1e8)
    group_valid = []
    for gid in range(num_groups):
        group_valid.append(torch.any((cgi == gid) & valid_candidate, dim=1))
    return torch.stack(group_valid, dim=1), valid_candidate


def masked_group_regret(group_index, batch, valid_candidate, device):
    cgi = batch["candidate_group_index"].to(device=device)
    oracle_score = batch["candidate_oracle_score"].to(device=device, dtype=torch.float32)
    masked_all = oracle_score.masked_fill(~valid_candidate, -1e9)
    best_score = masked_all.max(dim=1).values
    group_mask = valid_candidate & cgi.eq(group_index.unsqueeze(1))
    group_best = oracle_score.masked_fill(~group_mask, -1e9).max(dim=1).values
    regret = best_score - group_best
    return torch.where(torch.isfinite(regret), regret, torch.zeros_like(regret))


def baseline_group_prediction(baseline_model, batch, group_valid, device):
    if baseline_model is None:
        return None
    outputs = baseline_model(
        batch["front_rgb"].to(device=device, dtype=torch.float32),
        batch["wrist_rgb"].to(device=device, dtype=torch.float32),
        batch["wrist_depth"].to(device=device, dtype=torch.float32),
        batch["proprio"].to(device=device, dtype=torch.float32),
        batch["base_action"].to(device=device, dtype=torch.float32),
        batch["gripper_context"].to(device=device, dtype=torch.float32),
        batch["step_idx"].to(device=device),
        batch["candidate_actions_local"].to(device=device, dtype=torch.float32),
        phase_id=batch["phase_id"].to(device=device),
        phase_age=batch["phase_age"].to(device=device, dtype=torch.float32),
        steps_since_last_replan=batch["steps_since_last_replan"].to(device=device, dtype=torch.float32),
        current_delta_basin_target=batch["proxy_current_delta_basin_target"].to(device=device, dtype=torch.float32),
        current_dx_sign=batch["current_dx_sign"].to(device=device),
        current_dy_sign=batch["current_dy_sign"].to(device=device),
        current_dyaw_sign=batch["current_dyaw_sign"].to(device=device),
        basin_distance_bin=batch["basin_distance_bin"].to(device=device),
        candidate_mask=batch["candidate_mask"].to(device=device, dtype=torch.float32),
        return_aux=True,
    )
    logits = outputs["group_logits"]
    if logits.shape[1] != group_valid.shape[1]:
        common = min(logits.shape[1], group_valid.shape[1])
        logits = logits[:, :common]
        group_valid = group_valid[:, :common]
    return logits.masked_fill(~group_valid, -1e9).argmax(dim=-1)


def compute_loss(model, handoff_model, batch, device):
    handoff_latent = extract_handoff_latent(handoff_model, batch, device)
    logits = model(
        handoff_latent=handoff_latent,
        proxy_current_delta_basin_target=batch["proxy_current_delta_basin_target"].to(device=device, dtype=torch.float32),
        current_dx_sign=batch["current_dx_sign"].to(device=device),
        current_dy_sign=batch["current_dy_sign"].to(device=device),
        current_dyaw_sign=batch["current_dyaw_sign"].to(device=device),
        basin_distance_bin=batch["basin_distance_bin"].to(device=device),
    )
    group_valid, _ = compute_group_valid(batch, device, logits.shape[1])
    masked_logits = logits.masked_fill(~group_valid, -1e9)
    target = batch["best_group_index"].to(device=device)
    sample_weight = batch["sample_weight"].to(device=device, dtype=torch.float32)
    loss_vec = F.cross_entropy(masked_logits, target, reduction="none")
    loss = (loss_vec * sample_weight).sum() / torch.clamp(sample_weight.sum(), min=1e-6)
    with torch.no_grad():
        pred = masked_logits.argmax(dim=-1)
        acc = torch.mean((pred == target).float())
        xy_mask = batch["near_xy_hard"].to(device=device, dtype=torch.float32) > 0.5
        xy_acc = torch.mean((pred[xy_mask] == target[xy_mask]).float()) if torch.any(xy_mask) else torch.tensor(0.0, device=device)
    return loss, {"group_acc": float(acc.item()), "near_xy_group_acc": float(xy_acc.item())}


@torch.no_grad()
def evaluate(model, handoff_model, loader, device, baseline_model=None):
    model.eval()
    total = 0
    agg = {
        "loss": 0.0,
        "teacher_best_group_recall": 0.0,
        "near_xy_teacher_best_group_recall": 0.0,
        "near_yaw_teacher_best_group_recall": 0.0,
        "near_coupled_teacher_best_group_recall": 0.0,
        "ready_support_teacher_best_group_recall": 0.0,
        "group_regret_mean": 0.0,
        "group_regret_p95": 0.0,
        "baseline_teacher_best_group_recall": 0.0,
        "baseline_group_regret_mean": 0.0,
        "baseline_group_regret_p95": 0.0,
        "wrong_group_correction_rate": 0.0,
        "wrong_group_count": 0.0,
        "near_xy_count": 0.0,
        "near_yaw_count": 0.0,
        "near_coupled_count": 0.0,
        "ready_support_count": 0.0,
    }
    all_pred_ok, all_base_ok, all_pred_regret, all_base_regret = [], [], [], []
    all_near_xy, all_near_yaw, all_near_coupled, all_ready_support = [], [], [], []
    bucket_arrays: dict[str, list[np.ndarray]] = {
        "teacher_ready_v1": [],
        "xy_block_v1": [],
        "yaw_needed_v1": [],
        "far_negative_v1": [],
    }
    for batch in loader:
        loss, metrics = compute_loss(model, handoff_model, batch, device)
        bsz = batch["best_group_index"].shape[0]
        handoff_latent = extract_handoff_latent(handoff_model, batch, device)
        logits = model(
            handoff_latent=handoff_latent,
            proxy_current_delta_basin_target=batch["proxy_current_delta_basin_target"].to(device=device, dtype=torch.float32),
            current_dx_sign=batch["current_dx_sign"].to(device=device),
            current_dy_sign=batch["current_dy_sign"].to(device=device),
            current_dyaw_sign=batch["current_dyaw_sign"].to(device=device),
            basin_distance_bin=batch["basin_distance_bin"].to(device=device),
        )
        group_valid, valid_candidate = compute_group_valid(batch, device, logits.shape[1])
        pred = logits.masked_fill(~group_valid, -1e9).argmax(dim=-1)
        target = batch["best_group_index"].to(device=device)
        pred_ok = (pred == target).float()
        pred_regret = masked_group_regret(pred, batch, valid_candidate, device)
        base = baseline_group_prediction(baseline_model, batch, group_valid, device)
        if base is None:
            base_ok = torch.zeros_like(pred_ok)
            base_regret = torch.zeros_like(pred_regret)
            wrong_correction = torch.zeros_like(pred_ok)
            wrong_mask = torch.zeros_like(pred_ok, dtype=torch.bool)
        else:
            base_ok = (base == target).float()
            base_regret = masked_group_regret(base, batch, valid_candidate, device)
            wrong_mask = base != target
            wrong_correction = ((pred == target) & wrong_mask).float()
        total += bsz
        agg["loss"] += float(loss.item()) * bsz
        all_pred_ok.append(pred_ok.cpu().numpy())
        all_base_ok.append(base_ok.cpu().numpy())
        all_pred_regret.append(pred_regret.cpu().numpy())
        all_base_regret.append(base_regret.cpu().numpy())
        all_near_xy.append((batch["near_xy_hard"].numpy() > 0.5))
        all_near_yaw.append((batch["near_yaw_hard"].numpy() > 0.5))
        all_near_coupled.append((batch["near_coupled"].numpy() > 0.5))
        all_ready_support.append((batch["ready_support"].numpy() > 0.5) | (batch["teacher_truth_handoff_ready"].numpy() > 0.5))
        for key in bucket_arrays:
            bucket_arrays[key].append(batch[key].numpy() > 0.5)
        agg["wrong_group_count"] += float(wrong_mask.float().sum().item())
        agg["wrong_group_correction_rate"] += float(wrong_correction.sum().item())
    pred_ok = np.concatenate(all_pred_ok) if all_pred_ok else np.zeros((0,), dtype=np.float32)
    base_ok = np.concatenate(all_base_ok) if all_base_ok else np.zeros((0,), dtype=np.float32)
    pred_regret = np.concatenate(all_pred_regret) if all_pred_regret else np.zeros((0,), dtype=np.float32)
    base_regret = np.concatenate(all_base_regret) if all_base_regret else np.zeros((0,), dtype=np.float32)
    near_xy = np.concatenate(all_near_xy) if all_near_xy else np.zeros((0,), dtype=bool)
    near_yaw = np.concatenate(all_near_yaw) if all_near_yaw else np.zeros((0,), dtype=bool)
    near_coupled = np.concatenate(all_near_coupled) if all_near_coupled else np.zeros((0,), dtype=bool)
    ready_support = np.concatenate(all_ready_support) if all_ready_support else np.zeros((0,), dtype=bool)
    buckets = {
        key: (np.concatenate(vals) if vals else np.zeros((0,), dtype=bool))
        for key, vals in bucket_arrays.items()
    }

    def masked_mean(vals, mask=None):
        if mask is None:
            return float(np.mean(vals)) if vals.size else 0.0
        return float(np.mean(vals[mask])) if vals.size and np.any(mask) else 0.0

    agg["loss"] /= max(total, 1)
    agg["teacher_best_group_recall"] = masked_mean(pred_ok)
    agg["near_xy_teacher_best_group_recall"] = masked_mean(pred_ok, near_xy)
    agg["near_yaw_teacher_best_group_recall"] = masked_mean(pred_ok, near_yaw)
    agg["near_coupled_teacher_best_group_recall"] = masked_mean(pred_ok, near_coupled)
    agg["ready_support_teacher_best_group_recall"] = masked_mean(pred_ok, ready_support)
    agg["group_regret_mean"] = masked_mean(pred_regret)
    agg["group_regret_p95"] = float(np.percentile(pred_regret, 95)) if pred_regret.size else 0.0
    agg["baseline_teacher_best_group_recall"] = masked_mean(base_ok)
    agg["baseline_group_regret_mean"] = masked_mean(base_regret)
    agg["baseline_group_regret_p95"] = float(np.percentile(base_regret, 95)) if base_regret.size else 0.0
    wrong_count = max(agg["wrong_group_count"], 1.0)
    agg["wrong_group_correction_rate"] = agg["wrong_group_correction_rate"] / wrong_count
    agg["near_xy_count"] = int(np.sum(near_xy))
    agg["near_yaw_count"] = int(np.sum(near_yaw))
    agg["near_coupled_count"] = int(np.sum(near_coupled))
    agg["ready_support_count"] = int(np.sum(ready_support))
    agg["group_recall_gain_vs_baseline"] = agg["teacher_best_group_recall"] - agg["baseline_teacher_best_group_recall"]
    agg["group_regret_gain_vs_baseline"] = agg["baseline_group_regret_mean"] - agg["group_regret_mean"]
    regret_delta = base_regret - pred_regret
    for name, mask in buckets.items():
        prefix = name.replace("_v1", "")
        count = int(np.sum(mask))
        agg[f"{prefix}_count"] = count
        agg[f"{prefix}_low_confidence"] = bool(count < 25)
        agg[f"{prefix}_teacher_best_group_recall"] = masked_mean(pred_ok, mask)
        agg[f"{prefix}_baseline_teacher_best_group_recall"] = masked_mean(base_ok, mask)
        agg[f"{prefix}_group_regret_mean"] = masked_mean(pred_regret, mask)
        agg[f"{prefix}_baseline_group_regret_mean"] = masked_mean(base_regret, mask)
        agg[f"{prefix}_regret_delta_mean_baseline_minus_pred"] = masked_mean(regret_delta, mask)
    return agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_npz", required=True)
    ap.add_argument("--handoff_state_ckpt", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--val_ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--baseline_scorer_ckpt", default=None)
    ap.add_argument("--split_strategy", choices=("random", "episode_balanced"), default="random")
    ap.add_argument("--balanced_min_bucket_count", type=int, default=25)
    ap.add_argument("--val_episodes", default=None, help="comma-separated explicit validation episode ids")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dataset = NearReadyGroupDataset(args.dataset_npz)
    if args.val_episodes:
        train_idx, val_idx = split_by_explicit_val_episodes(dataset, args.val_episodes)
    elif args.split_strategy == "episode_balanced":
        train_idx, val_idx = split_by_episode_balanced(
            dataset,
            args.val_ratio,
            args.seed,
            min_bucket_count=args.balanced_min_bucket_count,
        )
    else:
        train_idx, val_idx = split_by_episode(dataset, args.val_ratio, args.seed)
    split_info = {
        "dataset_npz": args.dataset_npz,
        "seed": int(args.seed),
        "val_ratio": float(args.val_ratio),
        "split_strategy": args.split_strategy,
        "val_episodes": args.val_episodes,
        "train": split_bucket_summary(dataset, train_idx),
        "val": split_bucket_summary(dataset, val_idx),
    }
    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(Subset(dataset, val_idx), batch_size=args.batch_size, shuffle=False, num_workers=0)

    handoff_model = load_handoff_model(args.handoff_state_ckpt, device)
    baseline_model = load_pose_field_model(args.baseline_scorer_ckpt, device) if args.baseline_scorer_ckpt else None
    model = StudentGroupSelectorV2(
        num_groups=int(np.max(dataset.data["candidate_group_index"])) + 1,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "split_info.json").write_text(json.dumps(split_info, indent=2))
    print(json.dumps({"split_info": split_info}))
    history = []
    best = -float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        count = 0
        for batch in train_loader:
            loss, _ = compute_loss(model, handoff_model, batch, device)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            bsz = batch["best_group_index"].shape[0]
            train_loss += float(loss.item()) * bsz
            count += bsz
        metrics = evaluate(model, handoff_model, val_loader, device, baseline_model=baseline_model)
        row = {"epoch": epoch, "train_loss": train_loss / max(count, 1), **metrics}
        history.append(row)
        print(json.dumps(row))
        score = metrics["group_recall_gain_vs_baseline"] + 0.1 * metrics["group_regret_gain_vs_baseline"]
        if score > best:
            best = score
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "dataset_npz": args.dataset_npz,
                    "handoff_state_ckpt": args.handoff_state_ckpt,
                    "baseline_scorer_ckpt": args.baseline_scorer_ckpt,
                    "history": history,
                    "model_kind": "student_group_selector_v2",
                },
                output_dir / "student_group_selector_v2_best.pt",
            )
    (output_dir / "train_history.json").write_text(json.dumps(history, indent=2))


if __name__ == "__main__":
    main()
