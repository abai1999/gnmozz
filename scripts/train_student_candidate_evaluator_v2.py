from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset

from prismatic.models.student_candidate_evaluator_v2 import StudentCandidateEvaluatorV2
from prismatic.models.student_handoff_state_head_v2 import StudentHandoffStateHeadV2


class NearReadyCandidateDatasetV2(Dataset):
    def __init__(self, npz_path: str):
        arr = np.load(npz_path, allow_pickle=False)
        self.data = {k: np.asarray(arr[k]) for k in arr.files}
        self.length = int(self.data["best_candidate_index"].shape[0])
        if "episode_index" not in self.data:
            raise RuntimeError("dataset must contain episode_index")

    def __len__(self):
        return self.length

    def _substage_value(self, idx: int) -> int:
        if "substage_id" in self.data:
            return int(self.data["substage_id"][idx])
        return int(self.data.get("phase_id", np.ones((self.length,), dtype=np.int64))[idx])

    def __getitem__(self, idx):
        num_cands = int(self.data["candidate_mask"].shape[1])
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
            "gripper_context": torch.from_numpy(self.data["gripper_context"][idx].astype(np.float32)),
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
            "candidate_actions_local": torch.from_numpy(self.data["candidate_actions_local"][idx].astype(np.float32)),
            "candidate_group_index": torch.from_numpy(self.data["candidate_group_index"][idx].astype(np.int64)),
            "candidate_mask": torch.from_numpy(self.data["candidate_mask"][idx].astype(np.float32)),
            "candidate_oracle_score": torch.from_numpy(self.data["candidate_oracle_score"][idx].astype(np.float32)),
            "teacher_next_xy_norm_v1": torch.from_numpy(
                self.data.get("teacher_next_xy_norm_v1", np.zeros_like(self.data["candidate_oracle_score"]))[idx].astype(np.float32)
            ),
            "teacher_next_yaw_norm_v1": torch.from_numpy(
                self.data.get("teacher_next_yaw_norm_v1", np.zeros_like(self.data["candidate_oracle_score"]))[idx].astype(np.float32)
            ),
            "best_candidate_index": torch.tensor(int(self.data["best_candidate_index"][idx]), dtype=torch.long),
            "b2_yaw_aware_best_candidate_index_v3": torch.tensor(
                int(self.data.get("b2_yaw_aware_best_candidate_index_v3", self.data["best_candidate_index"])[idx]),
                dtype=torch.long,
            ),
            "b2_yaw_aware_candidate_scope_v3": torch.from_numpy(
                self.data.get("b2_yaw_aware_candidate_scope_v3", self.data["candidate_mask"])[idx].astype(np.float32)
            ),
            "b2_best_group_candidate_scope_v1": torch.from_numpy(
                self.data.get("b2_best_group_candidate_scope_v1", self.data["candidate_mask"])[idx].astype(np.float32)
            ),
            "b2_yaw_ladder_pair_mask_v5": torch.from_numpy(
                self.data.get(
                    "b2_yaw_ladder_pair_mask_v5",
                    np.zeros((self.length, 1, 1), dtype=np.float32),
                )[idx].astype(np.float32)
                if "b2_yaw_ladder_pair_mask_v5" in self.data
                else np.zeros((num_cands, num_cands), dtype=np.float32)
            ),
            "b2_yaw_ladder_pair_oracle_gap_v5": torch.from_numpy(
                self.data.get(
                    "b2_yaw_ladder_pair_oracle_gap_v5",
                    np.zeros((self.length, 1, 1), dtype=np.float32),
                )[idx].astype(np.float32)
                if "b2_yaw_ladder_pair_oracle_gap_v5" in self.data
                else np.zeros((num_cands, num_cands), dtype=np.float32)
            ),
            "b2_yaw_keep_pair_mask_v6": torch.from_numpy(
                self.data.get(
                    "b2_yaw_keep_pair_mask_v6",
                    np.zeros((self.length, 1, 1), dtype=np.float32),
                )[idx].astype(np.float32)
                if "b2_yaw_keep_pair_mask_v6" in self.data
                else np.zeros((num_cands, num_cands), dtype=np.float32)
            ),
            "b2_yaw_keep_pair_oracle_gap_v6": torch.from_numpy(
                self.data.get(
                    "b2_yaw_keep_pair_oracle_gap_v6",
                    np.zeros((self.length, 1, 1), dtype=np.float32),
                )[idx].astype(np.float32)
                if "b2_yaw_keep_pair_oracle_gap_v6" in self.data
                else np.zeros((num_cands, num_cands), dtype=np.float32)
            ),
            "best_group_index": torch.tensor(int(self.data["best_group_index"][idx]), dtype=torch.long),
            "sample_weight": torch.tensor(float(self.data.get("sample_weight", np.ones((self.length,), dtype=np.float32))[idx]), dtype=torch.float32),
            "near_coupled": torch.tensor(float(self.data.get("near_coupled", np.zeros((self.length,), dtype=np.float32))[idx]), dtype=torch.float32),
            "xy_block_v1": torch.tensor(float(self.data.get("xy_block_v1", np.zeros((self.length,), dtype=np.float32))[idx]), dtype=torch.float32),
            "yaw_needed_v1": torch.tensor(float(self.data.get("yaw_needed_v1", np.zeros((self.length,), dtype=np.float32))[idx]), dtype=torch.float32),
            "yaw_apply_v6": torch.tensor(float(self.data.get("yaw_apply_v6", np.zeros((self.length,), dtype=np.float32))[idx]), dtype=torch.float32),
            "yaw_keep_v6": torch.tensor(float(self.data.get("yaw_keep_v6", np.zeros((self.length,), dtype=np.float32))[idx]), dtype=torch.float32),
            "yaw_apply_v11": torch.tensor(float(self.data.get("yaw_apply_v11", self.data.get("yaw_apply_v6", np.zeros((self.length,), dtype=np.float32)))[idx]), dtype=torch.float32),
            "yaw_keep_v11": torch.tensor(float(self.data.get("yaw_keep_v11", self.data.get("yaw_keep_v6", np.zeros((self.length,), dtype=np.float32)))[idx]), dtype=torch.float32),
            "yaw_small_v11": torch.tensor(float(self.data.get("yaw_small_v11", np.zeros((self.length,), dtype=np.float32))[idx]), dtype=torch.float32),
            "yaw_mode_label_v7": torch.tensor(int(self.data.get("yaw_mode_label_v7", np.full((self.length,), -1, dtype=np.int64))[idx]), dtype=torch.long),
            "yaw_mode_margin_v7": torch.tensor(float(self.data.get("yaw_mode_margin_v7", np.zeros((self.length,), dtype=np.float32))[idx]), dtype=torch.float32),
            "yaw_mode_valid_v7": torch.tensor(float(self.data.get("yaw_mode_valid_v7", np.zeros((self.length,), dtype=np.float32))[idx]), dtype=torch.float32),
            "yaw_mode3_label_v11": torch.tensor(int(self.data.get("yaw_mode3_label_v11", np.full((self.length,), -1, dtype=np.int64))[idx]), dtype=torch.long),
            "yaw_mode_valid_v11": torch.tensor(float(self.data.get("yaw_mode_valid_v11", np.zeros((self.length,), dtype=np.float32))[idx]), dtype=torch.float32),
            "yaw_mode_confidence_v11": torch.tensor(float(self.data.get("yaw_mode_confidence_v11", np.ones((self.length,), dtype=np.float32))[idx]), dtype=torch.float32),
            "yaw_advantage_cont_v11": torch.tensor(float(self.data.get("yaw_advantage_cont_v11", np.zeros((self.length,), dtype=np.float32))[idx]), dtype=torch.float32),
            "far_negative_v1": torch.tensor(float(self.data.get("far_negative_v1", np.zeros((self.length,), dtype=np.float32))[idx]), dtype=torch.float32),
            "teacher_ready_v1": torch.tensor(float(self.data.get("teacher_ready_v1", np.zeros((self.length,), dtype=np.float32))[idx]), dtype=torch.float32),
            "episode_index": torch.tensor(int(self.data["episode_index"][idx]), dtype=torch.long),
        }


def split_by_episode(dataset: NearReadyCandidateDatasetV2, val_ratio: float, seed: int, min_val_yaw_needed_eps: int = 2):
    ep = np.asarray(dataset.data["episode_index"], dtype=np.int64)
    uniq = np.unique(ep)
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    val_n = max(1, int(round(uniq.size * val_ratio)))
    if val_n >= uniq.size:
        val_n = max(1, uniq.size - 1)
    yaw_needed = np.asarray(dataset.data.get("yaw_needed_v1", np.zeros((dataset.length,), dtype=np.float32))) > 0.5
    yaw_eps = np.unique(ep[yaw_needed])
    rng.shuffle(yaw_eps)
    val_list = []
    for e in yaw_eps.tolist():
        if len(val_list) >= min(int(min_val_yaw_needed_eps), val_n):
            break
        val_list.append(int(e))
    for e in uniq.tolist():
        if len(val_list) >= val_n:
            break
        if int(e) not in val_list:
            val_list.append(int(e))
    val_set = set(val_list)
    train_idx = [i for i, e in enumerate(ep.tolist()) if e not in val_set]
    val_idx = [i for i, e in enumerate(ep.tolist()) if e in val_set]
    return train_idx, val_idx


def split_by_yaw_mode_episode(
    dataset: NearReadyCandidateDatasetV2,
    val_ratio: float,
    seed: int,
    min_val_yaw_apply_eps: int = 2,
    min_val_yaw_keep_eps: int = 3,
    min_train_yaw_apply_eps: int = 2,
    min_train_yaw_keep_eps: int = 3,
):
    ep = np.asarray(dataset.data["episode_index"], dtype=np.int64)
    uniq = np.unique(ep)
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    val_n = max(1, int(round(uniq.size * val_ratio)))
    if val_n >= uniq.size:
        val_n = max(1, uniq.size - 1)
    yaw_apply = np.asarray(dataset.data.get("yaw_apply_v11", dataset.data.get("yaw_apply_v6", np.zeros((dataset.length,), dtype=np.float32)))) > 0.5
    yaw_keep = np.asarray(dataset.data.get("yaw_keep_v11", dataset.data.get("yaw_keep_v6", np.zeros((dataset.length,), dtype=np.float32)))) > 0.5
    apply_eps = np.unique(ep[yaw_apply])
    keep_eps = np.unique(ep[yaw_keep])
    rng.shuffle(apply_eps)
    rng.shuffle(keep_eps)
    val_list: list[int] = []
    apply_ep_set = set(int(x) for x in apply_eps.tolist())
    keep_ep_set = set(int(x) for x in keep_eps.tolist())

    def val_apply_count() -> int:
        return sum(1 for x in val_list if int(x) in apply_ep_set)

    def val_keep_count() -> int:
        return sum(1 for x in val_list if int(x) in keep_ep_set)

    def add_eps(candidates, need, kind: str):
        for e in candidates.tolist():
            if kind == "apply" and val_apply_count() >= need:
                break
            if kind == "keep" and val_keep_count() >= need:
                break
            if len(val_list) >= val_n:
                break
            if kind != "apply" and int(e) in apply_ep_set and val_apply_count() >= val_apply_need:
                continue
            if int(e) not in val_list:
                val_list.append(int(e))

    val_apply_need = min(
        int(min_val_yaw_apply_eps),
        max(int(len(apply_eps) - int(min_train_yaw_apply_eps)), 0),
        val_n,
    )
    val_keep_need = min(
        int(min_val_yaw_keep_eps),
        max(int(len(keep_eps) - int(min_train_yaw_keep_eps)), 0),
        val_n,
    )
    add_eps(apply_eps, val_apply_need, "apply")
    add_eps(keep_eps, val_keep_need, "keep")
    for e in uniq.tolist():
        if len(val_list) >= val_n:
            break
        if int(e) in apply_ep_set and val_apply_count() >= val_apply_need:
            continue
        if int(e) not in val_list:
            val_list.append(int(e))
    val_set = set(val_list)
    train_idx = [i for i, e in enumerate(ep.tolist()) if e not in val_set]
    val_idx = [i for i, e in enumerate(ep.tolist()) if e in val_set]
    split_info = {
        "train_episodes": int(len(set(ep[train_idx].tolist()))),
        "val_episodes": int(len(val_set)),
        "val_episode_indices": sorted(int(x) for x in val_set),
        "train_yaw_apply_eps": int(np.unique(ep[train_idx][yaw_apply[train_idx]]).size) if train_idx else 0,
        "val_yaw_apply_eps": int(np.unique(ep[val_idx][yaw_apply[val_idx]]).size) if val_idx else 0,
        "train_yaw_keep_eps": int(np.unique(ep[train_idx][yaw_keep[train_idx]]).size) if train_idx else 0,
        "val_yaw_keep_eps": int(np.unique(ep[val_idx][yaw_keep[val_idx]]).size) if val_idx else 0,
        "yaw_apply_eps_total": int(apply_eps.size),
        "yaw_keep_eps_total": int(keep_eps.size),
        "min_train_yaw_apply_eps": int(min_train_yaw_apply_eps),
        "min_train_yaw_keep_eps": int(min_train_yaw_keep_eps),
        "diagnostic_only": bool(apply_eps.size < 6 or keep_eps.size < 6),
    }
    return train_idx, val_idx, split_info


def build_yaw_mode_balanced_indices(
    dataset: NearReadyCandidateDatasetV2,
    base_indices: list[int],
    seed: int,
    samples_per_episode: int = 96,
    apply_fraction: float = 0.5,
) -> tuple[list[int], dict]:
    """Build a replacement-sampled mode-training index list balanced by episode.

    The ordinary shuffled loader can still be dominated by a few large keep/apply
    episodes. For Stage-M we instead give every apply episode and every keep
    episode a comparable voice, then let class-balanced CE handle row imbalance
    inside each batch.
    """
    ep = np.asarray(dataset.data["episode_index"], dtype=np.int64)
    label_key = "yaw_mode3_label_v11" if "yaw_mode3_label_v11" in dataset.data else "yaw_mode_label_v7"
    valid_key = "yaw_mode_valid_v11" if "yaw_mode_valid_v11" in dataset.data else "yaw_mode_valid_v7"
    label = np.asarray(dataset.data.get(label_key, np.full((dataset.length,), -1, dtype=np.int64)), dtype=np.int64)
    valid = np.asarray(dataset.data.get(valid_key, np.zeros((dataset.length,), dtype=np.float32)), dtype=np.float32) > 0.5
    base = np.asarray(base_indices, dtype=np.int64)
    rng = np.random.default_rng(seed)

    apply_label = int(np.max(label[valid])) if np.any(valid) else 1

    def rows_by_episode(mode_value: int) -> dict[int, np.ndarray]:
        mask = valid[base] & (label[base] == int(mode_value))
        out: dict[int, np.ndarray] = {}
        for e in np.unique(ep[base][mask]):
            rows = base[mask & (ep[base] == int(e))]
            if rows.size:
                out[int(e)] = rows
        return out

    keep_by_ep = rows_by_episode(0)
    apply_by_ep = rows_by_episode(apply_label)
    if not keep_by_ep or not apply_by_ep:
        return base_indices, {
            "enabled": False,
            "reason": "missing apply or keep episode",
            "apply_episode_count": len(apply_by_ep),
            "keep_episode_count": len(keep_by_ep),
        }

    samples_per_episode = max(int(samples_per_episode), 1)
    apply_fraction = float(np.clip(apply_fraction, 0.05, 0.95))
    total_target = samples_per_episode * (len(apply_by_ep) + len(keep_by_ep))
    apply_total = max(int(round(total_target * apply_fraction)), len(apply_by_ep))
    keep_total = max(total_target - apply_total, len(keep_by_ep))
    apply_per_ep = max(int(np.ceil(apply_total / len(apply_by_ep))), 1)
    keep_per_ep = max(int(np.ceil(keep_total / len(keep_by_ep))), 1)

    sampled: list[int] = []
    for rows in apply_by_ep.values():
        sampled.extend(rng.choice(rows, size=apply_per_ep, replace=rows.size < apply_per_ep).astype(int).tolist())
    for rows in keep_by_ep.values():
        sampled.extend(rng.choice(rows, size=keep_per_ep, replace=rows.size < keep_per_ep).astype(int).tolist())
    rng.shuffle(sampled)
    return sampled, {
        "enabled": True,
        "apply_episode_count": len(apply_by_ep),
        "keep_episode_count": len(keep_by_ep),
        "apply_rows_source": int(sum(v.size for v in apply_by_ep.values())),
        "keep_rows_source": int(sum(v.size for v in keep_by_ep.values())),
        "apply_rows_sampled": int(apply_per_ep * len(apply_by_ep)),
        "keep_rows_sampled": int(keep_per_ep * len(keep_by_ep)),
        "samples_per_episode": int(samples_per_episode),
        "apply_fraction": float(apply_fraction),
        "apply_label": int(apply_label),
    }


def load_handoff_model(ckpt_path: str, device):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model = StudentHandoffStateHeadV2().to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
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


def _candidate_scope(batch, device, mode: str = "best_group"):
    cmask = batch["candidate_mask"].to(device=device) > 0.5
    if mode == "yaw_aware":
        return (batch["b2_yaw_aware_candidate_scope_v3"].to(device=device, dtype=torch.float32) > 0.5) & cmask
    if mode == "best_group_field":
        return (batch["b2_best_group_candidate_scope_v1"].to(device=device, dtype=torch.float32) > 0.5) & cmask
    cgi = batch["candidate_group_index"].to(device=device)
    target_group = batch["best_group_index"].to(device=device)
    return cgi.eq(target_group.unsqueeze(1)) & cmask


def _target_candidate(batch, device, mode: str = "best_group"):
    if mode == "yaw_aware":
        return batch["b2_yaw_aware_best_candidate_index_v3"].to(device=device)
    return batch["best_candidate_index"].to(device=device)


def _yaw_mode_targets(batch, device):
    if "yaw_mode3_label_v11" in batch and "yaw_mode_valid_v11" in batch:
        label = batch["yaw_mode3_label_v11"].to(device=device)
        valid = (batch["yaw_mode_valid_v11"].to(device=device, dtype=torch.float32) > 0.5) & (label >= 0)
        target = torch.clamp(label, min=0).long()
        return target, valid
    if "yaw_mode_label_v7" in batch and "yaw_mode_valid_v7" in batch:
        label = batch["yaw_mode_label_v7"].to(device=device)
        valid = (batch["yaw_mode_valid_v7"].to(device=device, dtype=torch.float32) > 0.5) & (label >= 0)
        target = torch.clamp(label, min=0).long()
        return target, valid
    yaw_apply = batch["yaw_apply_v6"].to(device=device, dtype=torch.float32) > 0.5
    yaw_keep = batch["yaw_keep_v6"].to(device=device, dtype=torch.float32) > 0.5
    valid = yaw_apply | yaw_keep
    target = torch.zeros_like(batch["best_candidate_index"].to(device=device), dtype=torch.long)
    target[yaw_apply] = 1
    return target, valid


def _mode_gated_scope(batch, base_scope, device, mode_target=None, mode_logits=None, keep_yaw_abs: float = 0.035):
    candidate_actions = batch["candidate_actions_local"].to(device=device, dtype=torch.float32)
    small_yaw_scope = base_scope & (torch.abs(candidate_actions[:, :, 5]) <= float(keep_yaw_abs))
    if mode_target is None:
        if mode_logits is None:
            return base_scope
        mode_target = torch.argmax(mode_logits, dim=-1)
    keep_rows = mode_target == 0
    gated = base_scope.clone()
    has_small = torch.any(small_yaw_scope, dim=1)
    rows = keep_rows & has_small
    gated[rows] = small_yaw_scope[rows]
    return gated


def _scores_and_mode(model, latent, batch, device):
    if hasattr(model, "forward_with_mode"):
        out = model.forward_with_mode(
            handoff_latent=latent,
            proxy_current_delta_basin_target=batch["proxy_current_delta_basin_target"].to(device=device, dtype=torch.float32),
            candidate_actions_local=batch["candidate_actions_local"].to(device=device, dtype=torch.float32),
            candidate_mask=batch["candidate_mask"].to(device=device, dtype=torch.float32),
            yaw_aware_candidate_scope=batch["b2_yaw_aware_candidate_scope_v3"].to(device=device, dtype=torch.float32),
        )
        return {
            "generic": out["candidate_scores"],
            "keep": out.get("candidate_scores_keep", out["candidate_scores"]),
            "apply": out.get("candidate_scores_apply", out["candidate_scores"]),
            "mode_logits": out["yaw_mode_logits"],
        }
    scores = model(
        handoff_latent=latent,
        proxy_current_delta_basin_target=batch["proxy_current_delta_basin_target"].to(device=device, dtype=torch.float32),
        candidate_actions_local=batch["candidate_actions_local"].to(device=device, dtype=torch.float32),
        candidate_mask=batch["candidate_mask"].to(device=device, dtype=torch.float32),
        yaw_aware_candidate_scope=batch["b2_yaw_aware_candidate_scope_v3"].to(device=device, dtype=torch.float32),
    )
    return {
        "generic": scores,
        "keep": scores,
        "apply": scores,
        "mode_logits": torch.zeros((scores.shape[0], 2), device=device, dtype=scores.dtype),
    }


def _select_mode_scores(score_dict, mode_target=None, mode_logits=None):
    generic = score_dict["generic"]
    if mode_target is None:
        if mode_logits is None:
            return generic
        mode_target = torch.argmax(mode_logits, dim=-1)
    # Binary datasets use 0=keep, 1=apply. V11 3-way datasets use
    # 0=keep, 1=small-yaw/ambiguous, 2=apply. Until a dedicated small-yaw
    # score head exists, small-yaw stays conservative and uses the keep head.
    apply_label = score_dict["mode_logits"].shape[-1] - 1 if "mode_logits" in score_dict else 1
    return torch.where((mode_target == int(apply_label)).unsqueeze(1), score_dict["apply"], score_dict["keep"])


def _align_pairwise_tensor(pair_tensor: torch.Tensor, target_size: int, fill_value) -> torch.Tensor:
    """Pad legacy 65x65 pairwise tensors to expanded yaw-probe candidate banks."""
    if pair_tensor.shape[-2:] == (target_size, target_size):
        return pair_tensor
    if pair_tensor.shape[-2] <= target_size and pair_tensor.shape[-1] <= target_size:
        out = torch.full(
            (pair_tensor.shape[0], target_size, target_size),
            fill_value,
            dtype=pair_tensor.dtype,
            device=pair_tensor.device,
        )
        out[:, : pair_tensor.shape[-2], : pair_tensor.shape[-1]] = pair_tensor
        return out
    return torch.full(
        (pair_tensor.shape[0], target_size, target_size),
        fill_value,
        dtype=pair_tensor.dtype,
        device=pair_tensor.device,
    )


def set_train_stage(model: StudentCandidateEvaluatorV2, stage: str) -> None:
    for p in model.parameters():
        p.requires_grad = False
    if stage == "mode":
        modules = ["delta_encoder", "candidate_summary_encoder", "summary_context_head", "context_head", "yaw_mode_head"]
    elif stage == "rank":
        modules = ["delta_encoder", "action_encoder", "keep_yaw_score_head", "apply_yaw_score_head"]
    elif stage == "joint":
        modules = [
            "delta_encoder",
            "action_encoder",
            "candidate_summary_encoder",
            "summary_context_head",
            "context_head",
            "yaw_mode_head",
            "keep_yaw_score_head",
            "apply_yaw_score_head",
        ]
    else:
        modules = ["delta_encoder", "action_encoder", "candidate_summary_encoder", "summary_context_head", "context_head", "yaw_mode_head", "keep_yaw_score_head", "apply_yaw_score_head", "score_head"]
    for name in modules:
        module = getattr(model, name, None)
        if module is None:
            continue
        for p in module.parameters():
            p.requires_grad = True


def _episode_balanced_weight(episode_index: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    weight = torch.ones_like(episode_index, dtype=torch.float32)
    if not torch.any(valid):
        return weight
    eps, counts = torch.unique(episode_index[valid], return_counts=True)
    mean_inv = torch.mean(1.0 / torch.clamp(counts.float(), min=1.0))
    for ep_value, count in zip(eps, counts):
        ep_weight = (1.0 / torch.clamp(count.float(), min=1.0)) / torch.clamp(mean_inv, min=1e-6)
        weight = torch.where(episode_index == ep_value, ep_weight, weight)
    return weight


def _mode_loss_weight_and_target(
    batch,
    device,
    yaw_mode_target: torch.Tensor,
    yaw_mode_valid: torch.Tensor,
    yaw_mode_logits: torch.Tensor,
    sample_weight: torch.Tensor,
    mode_weight_policy: str,
    mode_small_policy: str,
):
    num_classes = int(yaw_mode_logits.shape[-1])
    apply_label = num_classes - 1
    loss_target = yaw_mode_target.clone()
    if mode_small_policy == "keep" and num_classes >= 3:
        loss_target = torch.where(loss_target == 1, torch.zeros_like(loss_target), loss_target)
    valid_targets = loss_target[yaw_mode_valid]
    num_valid = torch.clamp(yaw_mode_valid.float().sum(), min=1.0)
    class_counts = torch.stack([
        torch.clamp((valid_targets == c).float().sum(), min=1.0)
        for c in range(num_classes)
    ])
    class_weight_table = num_valid / (float(num_classes) * class_counts)
    class_weight = class_weight_table[valid_targets]
    if "yaw_mode_confidence_v11" in batch:
        conf = batch["yaw_mode_confidence_v11"].to(device=device, dtype=torch.float32)[yaw_mode_valid]
        class_weight = class_weight * torch.clamp(conf, min=0.25, max=2.0)
    if mode_weight_policy == "legacy_sample_weight":
        mode_weight = sample_weight[yaw_mode_valid] * class_weight
    elif mode_weight_policy == "capped_sample_weight":
        mode_weight = torch.clamp(sample_weight[yaw_mode_valid], min=0.25, max=1.0) * class_weight
    else:
        ep_weight = _episode_balanced_weight(batch["episode_index"].to(device=device), yaw_mode_valid)[yaw_mode_valid]
        mode_weight = ep_weight * class_weight
    with torch.no_grad():
        eff_keep = mode_weight[valid_targets == 0].sum()
        eff_apply = mode_weight[valid_targets == apply_label].sum()
        eff_small = mode_weight[valid_targets == 1].sum() if num_classes >= 3 else torch.tensor(0.0, device=device)
    return loss_target, mode_weight, {
        "mode_weight_keep_sum": float(eff_keep.item()),
        "mode_weight_apply_sum": float(eff_apply.item()),
        "mode_weight_small_sum": float(eff_small.item()),
    }


def compute_loss(
    model,
    handoff_model,
    batch,
    device,
    candidate_scope: str = "best_group",
    yaw_pairwise_weight: float = 0.35,
    yaw_keep_pairwise_weight: float = 0.35,
    yaw_mode_weight: float = 0.35,
    yaw_mode_apply_margin_weight: float = 0.0,
    yaw_mode_apply_margin_cap: float = 4.0,
    keep_yaw_abs: float = 0.035,
    train_stage: str = "joint",
    joint_predicted_mode_mix: float = 0.0,
    mode_weight_policy: str = "legacy_sample_weight",
    mode_small_policy: str = "separate",
):
    latent = extract_handoff_latent(handoff_model, batch, device)
    score_dict = _scores_and_mode(model, latent, batch, device)
    yaw_mode_logits = score_dict["mode_logits"]
    within_group = _candidate_scope(batch, device, candidate_scope)
    yaw_mode_target, yaw_mode_valid = _yaw_mode_targets(batch, device)
    use_pred_mode = bool(train_stage == "joint" and float(joint_predicted_mode_mix) > 0.0)
    if use_pred_mode:
        pred_scores = _select_mode_scores(score_dict, mode_logits=yaw_mode_logits)
        oracle_scores = _select_mode_scores(score_dict, mode_target=yaw_mode_target)
        mix = float(joint_predicted_mode_mix)
        scores = oracle_scores * (1.0 - mix) + pred_scores * mix
    else:
        scores = _select_mode_scores(score_dict, mode_target=yaw_mode_target)
    within_group = _mode_gated_scope(
        batch,
        within_group,
        device,
        mode_target=yaw_mode_target,
        keep_yaw_abs=keep_yaw_abs,
    )
    best_candidate = _target_candidate(batch, device, candidate_scope)
    # Teacher-mode gating should guide the search space without masking the
    # supervised target out of the CE loss on borderline/noisy rows.
    row_idx = torch.arange(within_group.shape[0], device=device)
    within_group[row_idx, best_candidate] = True
    masked_scores = scores.masked_fill(~within_group, -1e9)
    sample_weight = batch["sample_weight"].to(device=device, dtype=torch.float32)
    loss_vec = F.cross_entropy(masked_scores, best_candidate, reduction="none")
    rank_loss = (loss_vec * sample_weight).sum() / torch.clamp(sample_weight.sum(), min=1e-6)
    loss = rank_loss
    if torch.any(yaw_mode_valid):
        mode_loss_target, mode_weight, mode_weight_stats = _mode_loss_weight_and_target(
            batch,
            device,
            yaw_mode_target,
            yaw_mode_valid,
            yaw_mode_logits,
            sample_weight,
            mode_weight_policy,
            mode_small_policy,
        )
        mode_loss_vec = F.cross_entropy(yaw_mode_logits[yaw_mode_valid], mode_loss_target[yaw_mode_valid], reduction="none")
        if float(yaw_mode_apply_margin_weight) > 0.0 and "yaw_mode_margin_v7" in batch:
            margin = batch["yaw_mode_margin_v7"].to(device=device, dtype=torch.float32)[yaw_mode_valid]
            if "yaw_advantage_cont_v11" in batch:
                margin = batch["yaw_advantage_cont_v11"].to(device=device, dtype=torch.float32)[yaw_mode_valid]
            margin_scale = torch.clamp(margin, min=0.0, max=float(yaw_mode_apply_margin_cap)) / max(float(yaw_mode_apply_margin_cap), 1e-6)
            apply_boost = 1.0 + float(yaw_mode_apply_margin_weight) * margin_scale
            mode_weight = torch.where(mode_loss_target[yaw_mode_valid] == (yaw_mode_logits.shape[-1] - 1), mode_weight * apply_boost, mode_weight)
        yaw_mode_loss = (mode_loss_vec * mode_weight).sum() / torch.clamp(mode_weight.sum(), min=1e-6)
        if train_stage != "rank":
            loss = loss + float(yaw_mode_weight) * yaw_mode_loss
    else:
        yaw_mode_loss = yaw_mode_logits.sum() * 0.0
        mode_weight_stats = {"mode_weight_keep_sum": 0.0, "mode_weight_apply_sum": 0.0, "mode_weight_small_sum": 0.0}
    if train_stage == "mode":
        loss = float(yaw_mode_weight) * yaw_mode_loss
        yaw_pairwise_loss = torch.tensor(0.0, device=device)
        yaw_keep_pairwise_loss = torch.tensor(0.0, device=device)
        with torch.no_grad():
            pred = masked_scores.argmax(dim=-1)
            acc = torch.mean((pred == best_candidate).float())
            coupled = batch["near_coupled"].to(device=device, dtype=torch.float32) > 0.5
            coupled_acc = torch.mean((pred[coupled] == best_candidate[coupled]).float()) if torch.any(coupled) else torch.tensor(0.0, device=device)
            mode_pred = torch.argmax(yaw_mode_logits, dim=-1)
            yaw_mode_acc = torch.mean((mode_pred[yaw_mode_valid] == yaw_mode_target[yaw_mode_valid]).float()) if torch.any(yaw_mode_valid) else torch.tensor(0.0, device=device)
            yaw_mode_count = int(yaw_mode_valid.sum().item())
            yaw_mode_correct = int(((mode_pred[yaw_mode_valid] == yaw_mode_target[yaw_mode_valid]).sum()).item()) if torch.any(yaw_mode_valid) else 0
        return loss, {
            "candidate_acc": float(acc.item()),
            "near_coupled_acc": float(coupled_acc.item()),
            "yaw_pairwise_loss": 0.0,
            "yaw_keep_pairwise_loss": 0.0,
            "yaw_mode_loss": float(yaw_mode_loss.item()),
            "yaw_mode_acc": float(yaw_mode_acc.item()),
            "yaw_mode_count": yaw_mode_count,
            "yaw_mode_correct": yaw_mode_correct,
            **mode_weight_stats,
        }
    yaw_pair_mask = batch["b2_yaw_ladder_pair_mask_v5"].to(device=device, dtype=torch.float32) > 0.5
    yaw_pair_gap = batch["b2_yaw_ladder_pair_oracle_gap_v5"].to(device=device, dtype=torch.float32)
    if yaw_pair_mask.shape[-2:] != scores.shape[-1:] * 2:
        target_size = int(scores.shape[1])
        yaw_pair_mask = _align_pairwise_tensor(yaw_pair_mask, target_size, False)
        yaw_pair_gap = _align_pairwise_tensor(yaw_pair_gap, target_size, 0.0)
    if torch.any(yaw_pair_mask):
        apply_scores = score_dict["apply"]
        score_diff = apply_scores.unsqueeze(2) - apply_scores.unsqueeze(1)
        pair_loss = F.softplus(-score_diff)
        pair_weight = torch.clamp(yaw_pair_gap, min=0.0)
        pair_loss = pair_loss * pair_weight
        denom = torch.clamp(pair_weight[yaw_pair_mask].sum(), min=1e-6)
        yaw_pairwise_loss = pair_loss[yaw_pair_mask].sum() / denom
        loss = loss + float(yaw_pairwise_weight) * yaw_pairwise_loss
    else:
        yaw_pairwise_loss = torch.tensor(0.0, device=device)
    yaw_keep_pair_mask = batch["b2_yaw_keep_pair_mask_v6"].to(device=device, dtype=torch.float32) > 0.5
    yaw_keep_pair_gap = batch["b2_yaw_keep_pair_oracle_gap_v6"].to(device=device, dtype=torch.float32)
    if yaw_keep_pair_mask.shape[-2:] != scores.shape[-1:] * 2:
        target_size = int(scores.shape[1])
        yaw_keep_pair_mask = _align_pairwise_tensor(yaw_keep_pair_mask, target_size, False)
        yaw_keep_pair_gap = _align_pairwise_tensor(yaw_keep_pair_gap, target_size, 0.0)
    if torch.any(yaw_keep_pair_mask):
        keep_scores = score_dict["keep"]
        score_diff = keep_scores.unsqueeze(2) - keep_scores.unsqueeze(1)
        keep_pair_loss = F.softplus(-score_diff)
        keep_pair_weight = torch.clamp(yaw_keep_pair_gap, min=0.0)
        keep_pair_loss = keep_pair_loss * keep_pair_weight
        denom = torch.clamp(keep_pair_weight[yaw_keep_pair_mask].sum(), min=1e-6)
        yaw_keep_pairwise_loss = keep_pair_loss[yaw_keep_pair_mask].sum() / denom
        loss = loss + float(yaw_keep_pairwise_weight) * yaw_keep_pairwise_loss
    else:
        yaw_keep_pairwise_loss = torch.tensor(0.0, device=device)
    with torch.no_grad():
        pred = masked_scores.argmax(dim=-1)
        acc = torch.mean((pred == best_candidate).float())
        coupled = batch["near_coupled"].to(device=device, dtype=torch.float32) > 0.5
        coupled_acc = torch.mean((pred[coupled] == best_candidate[coupled]).float()) if torch.any(coupled) else torch.tensor(0.0, device=device)
        mode_pred = torch.argmax(yaw_mode_logits, dim=-1)
        yaw_mode_acc = torch.mean((mode_pred[yaw_mode_valid] == yaw_mode_target[yaw_mode_valid]).float()) if torch.any(yaw_mode_valid) else torch.tensor(0.0, device=device)
        yaw_mode_count = int(yaw_mode_valid.sum().item())
        yaw_mode_correct = int(((mode_pred[yaw_mode_valid] == yaw_mode_target[yaw_mode_valid]).sum()).item()) if torch.any(yaw_mode_valid) else 0
    return loss, {
        "candidate_acc": float(acc.item()),
        "near_coupled_acc": float(coupled_acc.item()),
        "yaw_pairwise_loss": float(yaw_pairwise_loss.item()),
        "yaw_keep_pairwise_loss": float(yaw_keep_pairwise_loss.item()),
        "yaw_mode_loss": float(yaw_mode_loss.item()),
        "yaw_mode_acc": float(yaw_mode_acc.item()),
        "yaw_mode_count": yaw_mode_count,
        "yaw_mode_correct": yaw_mode_correct,
        **mode_weight_stats,
    }


@torch.no_grad()
def evaluate(
    model,
    handoff_model,
    loader,
    device,
    candidate_scope: str = "best_group",
    yaw_pairwise_weight: float = 0.35,
    yaw_keep_pairwise_weight: float = 0.35,
    yaw_mode_weight: float = 0.35,
    yaw_mode_apply_margin_weight: float = 0.0,
    yaw_mode_apply_margin_cap: float = 4.0,
    keep_yaw_abs: float = 0.035,
    mode_eval: str = "predicted",
    mode_weight_policy: str = "legacy_sample_weight",
    mode_small_policy: str = "separate",
):
    model.eval()
    total = 0
    yaw_mode_total = 0
    yaw_mode_correct = 0
    agg = {
        "loss": 0.0,
        "candidate_acc": 0.0,
        "near_coupled_acc": 0.0,
        "yaw_pairwise_loss": 0.0,
        "yaw_keep_pairwise_loss": 0.0,
        "yaw_mode_loss": 0.0,
        "yaw_mode_acc": 0.0,
        "mode_weight_keep_sum": 0.0,
        "mode_weight_apply_sum": 0.0,
        "mode_weight_small_sum": 0.0,
    }
    pred_ok_all, regret_all, pairwise_all = [], [], []
    xy_gain_all, yaw_gain_all = [], []
    buckets = {"teacher_ready": [], "xy_block": [], "yaw_needed": [], "yaw_apply": [], "yaw_keep": [], "yaw_small": [], "far_negative": []}
    mode_targets_all, mode_preds_all, mode_valid_all = [], [], []
    for batch in loader:
        loss, metrics = compute_loss(
            model,
            handoff_model,
            batch,
            device,
            candidate_scope=candidate_scope,
            yaw_pairwise_weight=yaw_pairwise_weight,
            yaw_keep_pairwise_weight=yaw_keep_pairwise_weight,
            yaw_mode_weight=yaw_mode_weight,
            yaw_mode_apply_margin_weight=yaw_mode_apply_margin_weight,
            yaw_mode_apply_margin_cap=yaw_mode_apply_margin_cap,
            keep_yaw_abs=keep_yaw_abs,
            train_stage="joint",
            joint_predicted_mode_mix=0.0,
            mode_weight_policy=mode_weight_policy,
            mode_small_policy=mode_small_policy,
        )
        bsz = batch["best_candidate_index"].shape[0]
        total += bsz
        agg["loss"] += float(loss.item()) * bsz
        agg["candidate_acc"] += metrics["candidate_acc"] * bsz
        agg["near_coupled_acc"] += metrics["near_coupled_acc"] * bsz
        agg["yaw_pairwise_loss"] += metrics["yaw_pairwise_loss"] * bsz
        agg["yaw_keep_pairwise_loss"] += metrics["yaw_keep_pairwise_loss"] * bsz
        agg["yaw_mode_loss"] += metrics["yaw_mode_loss"] * bsz
        agg["mode_weight_keep_sum"] += metrics.get("mode_weight_keep_sum", 0.0)
        agg["mode_weight_apply_sum"] += metrics.get("mode_weight_apply_sum", 0.0)
        agg["mode_weight_small_sum"] += metrics.get("mode_weight_small_sum", 0.0)
        yaw_mode_total += int(metrics.get("yaw_mode_count", 0))
        yaw_mode_correct += int(metrics.get("yaw_mode_correct", 0))
        latent = extract_handoff_latent(handoff_model, batch, device)
        score_dict = _scores_and_mode(model, latent, batch, device)
        yaw_mode_logits = score_dict["mode_logits"]
        yaw_mode_target, yaw_mode_valid = _yaw_mode_targets(batch, device)
        mode_pred = torch.argmax(yaw_mode_logits, dim=-1)
        if mode_eval == "oracle":
            scores = _select_mode_scores(score_dict, mode_target=yaw_mode_target)
            gate_target = yaw_mode_target
        else:
            scores = _select_mode_scores(score_dict, mode_logits=yaw_mode_logits)
            gate_target = None
        within_group = _candidate_scope(batch, device, candidate_scope)
        within_group = _mode_gated_scope(
            batch,
            within_group,
            device,
            mode_target=gate_target,
            mode_logits=None if gate_target is not None else yaw_mode_logits,
            keep_yaw_abs=keep_yaw_abs,
        )
        masked_scores = scores.masked_fill(~within_group, -1e9)
        pred = masked_scores.argmax(dim=-1)
        best_candidate = _target_candidate(batch, device, candidate_scope)
        oracle = batch["candidate_oracle_score"].to(device=device, dtype=torch.float32)
        pred_score = oracle.gather(1, pred.unsqueeze(1)).squeeze(1)
        best_score = oracle.gather(1, best_candidate.unsqueeze(1)).squeeze(1)
        regret = best_score - pred_score
        pred_ok_all.append((pred == best_candidate).cpu().numpy())
        regret_all.append(regret.cpu().numpy())
        # Pairwise proxy: fraction of valid within-group candidates no better than the predicted candidate.
        valid_oracle = oracle.masked_fill(~within_group, -1e9)
        pairwise = (pred_score.unsqueeze(1) >= valid_oracle).float().masked_fill(~within_group, 0.0)
        denom = torch.clamp(within_group.float().sum(dim=1), min=1.0)
        pairwise_all.append((pairwise.sum(dim=1) / denom).cpu().numpy())
        next_xy = batch["teacher_next_xy_norm_v1"].to(device=device, dtype=torch.float32)
        next_yaw = batch["teacher_next_yaw_norm_v1"].to(device=device, dtype=torch.float32)
        best_xy = next_xy.gather(1, best_candidate.unsqueeze(1)).squeeze(1)
        pred_xy = next_xy.gather(1, pred.unsqueeze(1)).squeeze(1)
        best_yaw = next_yaw.gather(1, best_candidate.unsqueeze(1)).squeeze(1)
        pred_yaw = next_yaw.gather(1, pred.unsqueeze(1)).squeeze(1)
        xy_gain_all.append((best_xy - pred_xy).cpu().numpy())
        yaw_gain_all.append((best_yaw - pred_yaw).cpu().numpy())
        buckets["teacher_ready"].append(batch["teacher_ready_v1"].numpy() > 0.5)
        buckets["xy_block"].append(batch["xy_block_v1"].numpy() > 0.5)
        buckets["yaw_needed"].append(batch["yaw_needed_v1"].numpy() > 0.5)
        buckets["yaw_apply"].append(batch["yaw_apply_v11"].numpy() > 0.5)
        buckets["yaw_keep"].append(batch["yaw_keep_v11"].numpy() > 0.5)
        buckets["yaw_small"].append(batch["yaw_small_v11"].numpy() > 0.5)
        buckets["far_negative"].append(batch["far_negative_v1"].numpy() > 0.5)
        mode_targets_all.append(yaw_mode_target.cpu().numpy())
        mode_preds_all.append(mode_pred.cpu().numpy())
        mode_valid_all.append(yaw_mode_valid.cpu().numpy())
    for k in agg:
        agg[k] /= max(total, 1)
    agg["yaw_mode_acc"] = float(yaw_mode_correct / max(yaw_mode_total, 1))
    agg["yaw_mode_eval_count"] = int(yaw_mode_total)
    mode_targets = np.concatenate(mode_targets_all) if mode_targets_all else np.zeros((0,), dtype=np.int64)
    mode_preds = np.concatenate(mode_preds_all) if mode_preds_all else np.zeros((0,), dtype=np.int64)
    mode_valid = np.concatenate(mode_valid_all) if mode_valid_all else np.zeros((0,), dtype=bool)
    apply_label = int(np.max(mode_targets[mode_valid])) if np.any(mode_valid) else 1
    apply_mask = mode_valid & (mode_targets == apply_label)
    keep_mask = mode_valid & (mode_targets == 0)
    small_mask = mode_valid & (mode_targets == 1) if apply_label >= 2 else np.zeros_like(mode_valid, dtype=bool)
    agg["mode_apply_recall"] = float(np.mean(mode_preds[apply_mask] == 1)) if np.any(apply_mask) else 0.0
    if apply_label >= 2:
        agg["mode_apply_recall"] = float(np.mean(mode_preds[apply_mask] == apply_label)) if np.any(apply_mask) else 0.0
    agg["mode_keep_recall"] = float(np.mean(mode_preds[keep_mask] == 0)) if np.any(keep_mask) else 0.0
    agg["mode_small_recall"] = float(np.mean(mode_preds[small_mask] == 1)) if np.any(small_mask) else 0.0
    if apply_label >= 2 and np.any(small_mask):
        agg["mode_balanced_acc"] = float((agg["mode_apply_recall"] + agg["mode_keep_recall"] + agg["mode_small_recall"]) / 3.0)
    else:
        agg["mode_balanced_acc"] = float((agg["mode_apply_recall"] + agg["mode_keep_recall"]) * 0.5) if np.any(mode_valid) else 0.0
    binary_mask = mode_valid & ((mode_targets == 0) | (mode_targets == apply_label))
    binary_apply_mask = mode_valid & (mode_targets == apply_label)
    binary_keep_mask = mode_valid & (mode_targets == 0)
    agg["mode_apply_recall_binary"] = float(np.mean(mode_preds[binary_apply_mask] == apply_label)) if np.any(binary_apply_mask) else 0.0
    agg["mode_keep_recall_binary"] = float(np.mean(mode_preds[binary_keep_mask] == 0)) if np.any(binary_keep_mask) else 0.0
    agg["mode_apply_keep_balanced_acc"] = (
        float((agg["mode_apply_recall_binary"] + agg["mode_keep_recall_binary"]) * 0.5)
        if np.any(binary_mask)
        else 0.0
    )
    agg["mode_binary_eval_count"] = int(np.sum(binary_mask))
    pred_ok = np.concatenate(pred_ok_all) if pred_ok_all else np.zeros((0,), dtype=bool)
    regret = np.concatenate(regret_all) if regret_all else np.zeros((0,), dtype=np.float32)
    pairwise = np.concatenate(pairwise_all) if pairwise_all else np.zeros((0,), dtype=np.float32)
    xy_gain = np.concatenate(xy_gain_all) if xy_gain_all else np.zeros((0,), dtype=np.float32)
    yaw_gain = np.concatenate(yaw_gain_all) if yaw_gain_all else np.zeros((0,), dtype=np.float32)

    def mean(vals, mask=None):
        if mask is None:
            return float(np.mean(vals)) if vals.size else 0.0
        return float(np.mean(vals[mask])) if vals.size and np.any(mask) else 0.0

    agg["candidate_regret_mean"] = mean(regret)
    agg["candidate_regret_p95"] = float(np.percentile(regret, 95)) if regret.size else 0.0
    agg["within_group_pairwise_accuracy"] = mean(pairwise)
    agg["xy_next_norm_gap_best_minus_pred_mean"] = mean(xy_gain)
    agg["yaw_next_norm_gap_best_minus_pred_mean"] = mean(yaw_gain)
    for name, parts in buckets.items():
        mask = np.concatenate(parts) if parts else np.zeros((0,), dtype=bool)
        agg[f"{name}_count"] = int(np.sum(mask))
        agg[f"{name}_low_confidence"] = bool(int(np.sum(mask)) < 25)
        agg[f"{name}_candidate_acc"] = mean(pred_ok.astype(np.float32), mask)
        agg[f"{name}_candidate_regret_mean"] = mean(regret, mask)
        agg[f"{name}_pairwise_accuracy"] = mean(pairwise, mask)
        agg[f"{name}_xy_next_norm_gap_best_minus_pred_mean"] = mean(xy_gain, mask)
        agg[f"{name}_yaw_next_norm_gap_best_minus_pred_mean"] = mean(yaw_gain, mask)
    return agg


def _mode_gate_pass(metrics: dict) -> bool:
    return (
        float(metrics.get("mode_apply_keep_balanced_acc", 0.0)) >= 0.65
        and float(metrics.get("mode_apply_recall_binary", 0.0)) >= 0.55
        and float(metrics.get("mode_keep_recall_binary", 0.0)) >= 0.65
    )


def _candidate_regret_gate_pass(predicted: dict, oracle: dict) -> bool:
    pred_yaw = float(predicted.get("yaw_needed_candidate_regret_mean", 1e9))
    oracle_yaw = float(oracle.get("yaw_needed_candidate_regret_mean", 1e9))
    return (
        pred_yaw <= max(oracle_yaw + 2.0, oracle_yaw * 1.35)
        and float(predicted.get("yaw_keep_candidate_regret_mean", 1e9)) <= 8.0
        and float(predicted.get("yaw_apply_candidate_regret_mean", 1e9)) <= 8.0
        and float(predicted.get("teacher_ready_candidate_regret_mean", 1e9)) <= 2.5
        and float(predicted.get("xy_block_candidate_regret_mean", 1e9)) <= 5.0
    )


def _build_gate_reports(history: list[dict]) -> tuple[dict, dict, dict]:
    if not history:
        blocked = {
            "decision": "offline_blocked",
            "reasons": ["no training history"],
            "mode_gate_pass": False,
            "candidate_regret_gate_pass": False,
        }
        return blocked, blocked, blocked
    best_mode_row = max(
        history,
        key=lambda r: (
            float(r.get("predicted_mode_apply_keep_balanced_acc", 0.0)),
            float(r.get("predicted_mode_apply_recall_binary", 0.0)),
            float(r.get("predicted_mode_keep_recall_binary", 0.0)),
        ),
    )
    rank_rows = [r for r in history if r.get("train_stage") == "rank"]
    best_rank_row = min(
        rank_rows or history,
        key=lambda r: float(r.get("predicted_yaw_needed_candidate_regret_mean", 1e9)),
    )
    mode_metrics = {k[len("predicted_"):]: v for k, v in best_mode_row.items() if k.startswith("predicted_")}
    rank_pred = {k[len("predicted_"):]: v for k, v in best_rank_row.items() if k.startswith("predicted_")}
    rank_oracle = {k[len("oracle_"):]: v for k, v in best_rank_row.items() if k.startswith("oracle_")}
    mode_pass = _mode_gate_pass(mode_metrics)
    regret_pass = _candidate_regret_gate_pass(rank_pred, rank_oracle)
    mode_report = {
        "passes_gate": bool(mode_pass),
        "best_epoch": int(best_mode_row.get("epoch", -1)),
        "mode_apply_keep_balanced_acc": float(mode_metrics.get("mode_apply_keep_balanced_acc", 0.0)),
        "mode_apply_recall_binary": float(mode_metrics.get("mode_apply_recall_binary", 0.0)),
        "mode_keep_recall_binary": float(mode_metrics.get("mode_keep_recall_binary", 0.0)),
        "mode_binary_eval_count": int(mode_metrics.get("mode_binary_eval_count", 0)),
        "thresholds": {
            "mode_apply_keep_balanced_acc": 0.65,
            "mode_apply_recall_binary": 0.55,
            "mode_keep_recall_binary": 0.65,
        },
    }
    candidate_report = {
        "passes_gate": bool(regret_pass),
        "best_epoch": int(best_rank_row.get("epoch", -1)),
        "predicted_yaw_needed_candidate_regret_mean": float(rank_pred.get("yaw_needed_candidate_regret_mean", 0.0)),
        "oracle_yaw_needed_candidate_regret_mean": float(rank_oracle.get("yaw_needed_candidate_regret_mean", 0.0)),
        "predicted_yaw_keep_candidate_regret_mean": float(rank_pred.get("yaw_keep_candidate_regret_mean", 0.0)),
        "predicted_yaw_apply_candidate_regret_mean": float(rank_pred.get("yaw_apply_candidate_regret_mean", 0.0)),
        "predicted_teacher_ready_candidate_regret_mean": float(rank_pred.get("teacher_ready_candidate_regret_mean", 0.0)),
        "predicted_xy_block_candidate_regret_mean": float(rank_pred.get("xy_block_candidate_regret_mean", 0.0)),
    }
    reasons = []
    if not mode_pass:
        reasons.append("mode_gate_failed")
    if not regret_pass:
        reasons.append("candidate_regret_gate_failed")
    decision = {
        "decision": "offline_shadow_candidate" if mode_pass and regret_pass else "offline_blocked",
        "mode_gate_pass": bool(mode_pass),
        "candidate_regret_gate_pass": bool(regret_pass),
        "reasons": reasons,
        "best_mode_epoch": int(best_mode_row.get("epoch", -1)),
        "best_rank_epoch": int(best_rank_row.get("epoch", -1)),
        "runtime_shadow_allowed": bool(mode_pass and regret_pass),
    }
    return mode_report, candidate_report, decision


@torch.no_grad()
def build_focus_episode_diagnostics(
    model,
    handoff_model,
    dataset: NearReadyCandidateDatasetV2,
    episodes: list[int],
    device,
    candidate_scope: str,
    keep_yaw_abs: float,
    batch_size: int,
) -> dict:
    ep_arr = np.asarray(dataset.data["episode_index"], dtype=np.int64)
    groups = {f"ep{int(e):03d}": np.flatnonzero(ep_arr == int(e)).astype(int).tolist() for e in episodes}
    groups["focus_all"] = np.flatnonzero(np.isin(ep_arr, episodes)).astype(int).tolist()
    out = {}
    for name, indices in groups.items():
        if not indices:
            out[name] = {"rows": 0}
            continue
        loader = DataLoader(Subset(dataset, indices), batch_size=batch_size, shuffle=False, num_workers=0)
        target_parts, pred_mode_parts, valid_parts = [], [], []
        regret_parts, ok_parts = [], []
        bucket_parts = {"yaw_needed": [], "yaw_apply": [], "yaw_keep": [], "yaw_small": []}
        for batch in loader:
            latent = extract_handoff_latent(handoff_model, batch, device)
            score_dict = _scores_and_mode(model, latent, batch, device)
            logits = score_dict["mode_logits"]
            mode_target, mode_valid = _yaw_mode_targets(batch, device)
            mode_pred = torch.argmax(logits, dim=-1)
            scores = _select_mode_scores(score_dict, mode_logits=logits)
            scope = _candidate_scope(batch, device, candidate_scope)
            scope = _mode_gated_scope(batch, scope, device, mode_logits=logits, keep_yaw_abs=keep_yaw_abs)
            pred = scores.masked_fill(~scope, -1e9).argmax(dim=-1)
            best = _target_candidate(batch, device, candidate_scope)
            oracle = batch["candidate_oracle_score"].to(device=device, dtype=torch.float32)
            regret = oracle.gather(1, best.unsqueeze(1)).squeeze(1) - oracle.gather(1, pred.unsqueeze(1)).squeeze(1)
            target_parts.append(mode_target.cpu().numpy())
            pred_mode_parts.append(mode_pred.cpu().numpy())
            valid_parts.append(mode_valid.cpu().numpy().astype(bool))
            regret_parts.append(regret.cpu().numpy())
            ok_parts.append((pred == best).cpu().numpy())
            bucket_parts["yaw_needed"].append(batch["yaw_needed_v1"].numpy() > 0.5)
            bucket_parts["yaw_apply"].append(batch["yaw_apply_v11"].numpy() > 0.5)
            bucket_parts["yaw_keep"].append(batch["yaw_keep_v11"].numpy() > 0.5)
            bucket_parts["yaw_small"].append(batch["yaw_small_v11"].numpy() > 0.5)
        target = np.concatenate(target_parts)
        pred_mode = np.concatenate(pred_mode_parts)
        valid = np.concatenate(valid_parts)
        regret = np.concatenate(regret_parts)
        ok = np.concatenate(ok_parts)
        buckets = {k: np.concatenate(v) for k, v in bucket_parts.items()}
        apply_label = int(np.max(target[valid])) if np.any(valid) else 1
        keep_mask = valid & (target == 0)
        apply_mask = valid & (target == apply_label)

        def mean(vals, mask=None):
            if mask is None:
                return float(np.mean(vals)) if vals.size else 0.0
            return float(np.mean(vals[mask])) if vals.size and np.any(mask) else 0.0

        def counts(vals, mask):
            if not np.any(mask):
                return {}
            keys, cnt = np.unique(vals[mask], return_counts=True)
            return {str(int(k)): int(c) for k, c in zip(keys, cnt)}

        out[name] = {
            "rows": int(len(indices)),
            "mode_valid_rows": int(np.sum(valid)),
            "target_counts": counts(target, valid),
            "pred_mode_counts_on_valid": counts(pred_mode, valid),
            "mode_apply_recall_binary": float(np.mean(pred_mode[apply_mask] == apply_label)) if np.any(apply_mask) else 0.0,
            "mode_keep_recall_binary": float(np.mean(pred_mode[keep_mask] == 0)) if np.any(keep_mask) else 0.0,
            "candidate_acc": mean(ok.astype(np.float32)),
            "candidate_regret_mean": mean(regret),
            "yaw_needed_regret_mean": mean(regret, buckets["yaw_needed"]),
            "yaw_apply_regret_mean": mean(regret, buckets["yaw_apply"]),
            "yaw_keep_regret_mean": mean(regret, buckets["yaw_keep"]),
            "yaw_small_regret_mean": mean(regret, buckets["yaw_small"]),
            "yaw_needed_rows": int(np.sum(buckets["yaw_needed"])),
            "yaw_apply_rows": int(np.sum(buckets["yaw_apply"])),
            "yaw_keep_rows": int(np.sum(buckets["yaw_keep"])),
            "yaw_small_rows": int(np.sum(buckets["yaw_small"])),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_npz", required=True)
    ap.add_argument("--handoff_state_ckpt", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--b2_training_plan", default="legacy", choices=["legacy", "mode_rank_eval_v13"])
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--stage_m_epochs", type=int, default=2)
    ap.add_argument("--stage_r_epochs", type=int, default=4)
    ap.add_argument("--stage_j_epochs", type=int, default=2)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--val_ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--min_val_yaw_needed_eps", type=int, default=2)
    ap.add_argument("--min_val_yaw_apply_eps", type=int, default=2)
    ap.add_argument("--min_val_yaw_keep_eps", type=int, default=3)
    ap.add_argument("--min_train_yaw_apply_eps", type=int, default=2)
    ap.add_argument("--min_train_yaw_keep_eps", type=int, default=3)
    ap.add_argument("--yaw_mode_stratified_split", action="store_true")
    ap.add_argument("--yaw_mode_balanced_mode_stage", action="store_true")
    ap.add_argument("--yaw_mode_samples_per_episode", type=int, default=96)
    ap.add_argument("--yaw_mode_apply_fraction", type=float, default=0.5)
    ap.add_argument("--yaw_pairwise_weight", type=float, default=0.35)
    ap.add_argument("--yaw_keep_pairwise_weight", type=float, default=0.35)
    ap.add_argument("--yaw_mode_weight", type=float, default=0.35)
    ap.add_argument("--yaw_mode_apply_margin_weight", type=float, default=0.0)
    ap.add_argument("--yaw_mode_apply_margin_cap", type=float, default=4.0)
    ap.add_argument(
        "--mode_weight_policy",
        default="legacy_sample_weight",
        choices=["legacy_sample_weight", "capped_sample_weight", "episode_class_balanced_no_sample_weight"],
    )
    ap.add_argument("--mode_binary_gate_eval", action="store_true")
    ap.add_argument("--disable_joint_predicted_mode_training", action="store_true", default=False)
    ap.add_argument("--mode_small_policy", default="separate", choices=["separate", "keep"])
    ap.add_argument("--summary_first_mode_only", action="store_true")
    ap.add_argument("--focus_episodes", default="18,34,45")
    ap.add_argument(
        "--yaw_mode_num_classes",
        type=int,
        default=0,
        help="0 = infer from dataset; v7 datasets infer 2, v11 datasets infer 3.",
    )
    ap.add_argument("--yaw_keep_abs", type=float, default=0.035)
    ap.add_argument(
        "--candidate_scope",
        type=str,
        default="best_group",
        choices=["best_group", "best_group_field", "yaw_aware"],
        help="Candidate scope for B2 ranking. yaw_aware enables the v3 expanded scope on yaw-needed rows.",
    )
    args = ap.parse_args()
    if args.b2_training_plan == "mode_rank_eval_v13":
        if args.mode_weight_policy == "legacy_sample_weight":
            args.mode_weight_policy = "episode_class_balanced_no_sample_weight"
        if args.mode_small_policy == "separate":
            args.mode_small_policy = "keep"
        args.mode_binary_gate_eval = True
        args.disable_joint_predicted_mode_training = True

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dataset = NearReadyCandidateDatasetV2(args.dataset_npz)
    if args.yaw_mode_stratified_split:
        train_idx, val_idx, split_info = split_by_yaw_mode_episode(
            dataset,
            args.val_ratio,
            args.seed,
            min_val_yaw_apply_eps=args.min_val_yaw_apply_eps,
            min_val_yaw_keep_eps=args.min_val_yaw_keep_eps,
            min_train_yaw_apply_eps=args.min_train_yaw_apply_eps,
            min_train_yaw_keep_eps=args.min_train_yaw_keep_eps,
        )
    else:
        train_idx, val_idx = split_by_episode(dataset, args.val_ratio, args.seed, args.min_val_yaw_needed_eps)
        split_info = {"split": "episode", "diagnostic_only": False}
    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=args.batch_size, shuffle=True, num_workers=0)
    mode_train_idx = train_idx
    mode_sampler_info = {"enabled": False}
    if args.yaw_mode_balanced_mode_stage:
        mode_train_idx, mode_sampler_info = build_yaw_mode_balanced_indices(
            dataset,
            train_idx,
            seed=args.seed,
            samples_per_episode=args.yaw_mode_samples_per_episode,
            apply_fraction=args.yaw_mode_apply_fraction,
        )
    mode_train_loader = DataLoader(Subset(dataset, mode_train_idx), batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(Subset(dataset, val_idx), batch_size=args.batch_size, shuffle=False, num_workers=0)

    handoff_model = load_handoff_model(args.handoff_state_ckpt, device)
    yaw_mode_num_classes = int(args.yaw_mode_num_classes)
    if yaw_mode_num_classes <= 0:
        if "yaw_mode3_label_v11" in dataset.data:
            labels = np.asarray(dataset.data["yaw_mode3_label_v11"], dtype=np.int64)
            valid = np.asarray(dataset.data.get("yaw_mode_valid_v11", labels >= 0), dtype=np.float32) > 0.5
            yaw_mode_num_classes = int(max(2, np.max(labels[valid]) + 1 if np.any(valid) else 3))
        else:
            yaw_mode_num_classes = 2
    model = StudentCandidateEvaluatorV2(yaw_mode_classes=yaw_mode_num_classes).to(device)
    if hasattr(model, "set_mode_input_path"):
        model.set_mode_input_path("summary_only" if bool(args.summary_first_mode_only) else "hybrid")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    split_info["mode_sampler"] = mode_sampler_info
    (output_dir / "split_info.json").write_text(json.dumps(split_info, indent=2))
    history = []
    best = float("inf")
    stage_j_epochs = 0 if bool(args.disable_joint_predicted_mode_training) else int(args.stage_j_epochs)
    total_epochs = int(args.stage_m_epochs + args.stage_r_epochs + stage_j_epochs)
    if args.epochs != 8 and args.epochs != total_epochs:
        total_epochs = int(args.epochs)
    best_mode_metric = -float("inf")
    best_mode_state = None
    best_mode_epoch = -1
    best_rank_metric = float("inf")
    best_rank_state = None
    best_rank_epoch = -1
    restored_mode_for_rank = False
    for epoch in range(1, total_epochs + 1):
        if (
            args.b2_training_plan == "mode_rank_eval_v13"
            and not restored_mode_for_rank
            and epoch > int(args.stage_m_epochs)
            and best_mode_state is not None
        ):
            model.load_state_dict(best_mode_state, strict=False)
            restored_mode_for_rank = True
        if epoch <= args.stage_m_epochs:
            train_stage = "mode"
            predicted_mix = 0.0
        elif epoch <= args.stage_m_epochs + args.stage_r_epochs:
            train_stage = "rank"
            predicted_mix = 0.0
        else:
            train_stage = "joint"
            denom = max(int(stage_j_epochs), 1)
            predicted_mix = 0.0 if args.disable_joint_predicted_mode_training else min(0.5, 0.5 * (epoch - args.stage_m_epochs - args.stage_r_epochs) / denom)
        if hasattr(model, "set_mode_input_path"):
            model.set_mode_input_path("summary_only" if bool(args.summary_first_mode_only) else "hybrid")
        set_train_stage(model, train_stage)
        model.train()
        train_loss = 0.0
        count = 0
        active_train_loader = mode_train_loader if train_stage == "mode" and args.yaw_mode_balanced_mode_stage else train_loader
        for batch in active_train_loader:
            loss, _ = compute_loss(
                model,
                handoff_model,
                batch,
                device,
                candidate_scope=args.candidate_scope,
                yaw_pairwise_weight=args.yaw_pairwise_weight,
                yaw_keep_pairwise_weight=args.yaw_keep_pairwise_weight,
                yaw_mode_weight=args.yaw_mode_weight,
                yaw_mode_apply_margin_weight=args.yaw_mode_apply_margin_weight,
                yaw_mode_apply_margin_cap=args.yaw_mode_apply_margin_cap,
                keep_yaw_abs=args.yaw_keep_abs,
                train_stage=train_stage,
                joint_predicted_mode_mix=predicted_mix,
                mode_weight_policy=args.mode_weight_policy,
                mode_small_policy=args.mode_small_policy,
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            bsz = batch["best_candidate_index"].shape[0]
            train_loss += float(loss.item()) * bsz
            count += bsz
        oracle_metrics = evaluate(
            model,
            handoff_model,
            val_loader,
            device,
            candidate_scope=args.candidate_scope,
            yaw_pairwise_weight=args.yaw_pairwise_weight,
            yaw_keep_pairwise_weight=args.yaw_keep_pairwise_weight,
            yaw_mode_weight=args.yaw_mode_weight,
            yaw_mode_apply_margin_weight=args.yaw_mode_apply_margin_weight,
            yaw_mode_apply_margin_cap=args.yaw_mode_apply_margin_cap,
            keep_yaw_abs=args.yaw_keep_abs,
            mode_eval="oracle",
            mode_weight_policy=args.mode_weight_policy,
            mode_small_policy=args.mode_small_policy,
        )
        predicted_metrics = evaluate(
            model,
            handoff_model,
            val_loader,
            device,
            candidate_scope=args.candidate_scope,
            yaw_pairwise_weight=args.yaw_pairwise_weight,
            yaw_keep_pairwise_weight=args.yaw_keep_pairwise_weight,
            yaw_mode_weight=args.yaw_mode_weight,
            yaw_mode_apply_margin_weight=args.yaw_mode_apply_margin_weight,
            yaw_mode_apply_margin_cap=args.yaw_mode_apply_margin_cap,
            keep_yaw_abs=args.yaw_keep_abs,
            mode_eval="predicted",
            mode_weight_policy=args.mode_weight_policy,
            mode_small_policy=args.mode_small_policy,
        )
        metrics = {f"oracle_{k}": v for k, v in oracle_metrics.items()}
        metrics.update({f"predicted_{k}": v for k, v in predicted_metrics.items()})
        row = {
            "epoch": epoch,
            "train_stage": train_stage,
            "joint_predicted_mode_mix": predicted_mix,
            "train_loss": train_loss / max(count, 1),
            **metrics,
        }
        history.append(row)
        print(json.dumps(row))
        if train_stage == "mode":
            mode_metric = float(predicted_metrics.get("mode_apply_keep_balanced_acc", predicted_metrics.get("mode_balanced_acc", 0.0)))
            if mode_metric > best_mode_metric:
                best_mode_metric = mode_metric
                best_mode_epoch = int(epoch)
                best_mode_state = copy.deepcopy(model.state_dict())
                torch.save(
                    {
                        "model_state_dict": best_mode_state,
                        "dataset_npz": args.dataset_npz,
                        "handoff_state_ckpt": args.handoff_state_ckpt,
                        "candidate_scope": args.candidate_scope,
                        "yaw_mode_enabled": True,
                        "yaw_mode_num_classes": int(yaw_mode_num_classes),
                        "yaw_keep_abs": args.yaw_keep_abs,
                        "mode_feature_version": getattr(model, "mode_feature_version", "unknown"),
                        "split_info": split_info,
                        "training_schedule": {
                            "stage_m_epochs": args.stage_m_epochs,
                            "stage_r_epochs": args.stage_r_epochs,
                            "stage_j_epochs": stage_j_epochs,
                        },
                        "history": history,
                        "model_kind": "student_candidate_evaluator_v2",
                        "selected_epoch": int(epoch),
                        "selection_metric": "mode_apply_keep_balanced_acc",
                    },
                    output_dir / "student_candidate_evaluator_v2_best_mode.pt",
                )
        if train_stage == "rank":
            rank_metric = float(predicted_metrics.get("yaw_needed_candidate_regret_mean", predicted_metrics.get("candidate_regret_mean", 1e9)))
            if rank_metric < best_rank_metric:
                best_rank_metric = rank_metric
                best_rank_epoch = int(epoch)
                best_rank_state = copy.deepcopy(model.state_dict())
                torch.save(
                    {
                        "model_state_dict": best_rank_state,
                        "dataset_npz": args.dataset_npz,
                        "handoff_state_ckpt": args.handoff_state_ckpt,
                        "candidate_scope": args.candidate_scope,
                        "yaw_mode_enabled": True,
                        "yaw_mode_num_classes": int(yaw_mode_num_classes),
                        "yaw_keep_abs": args.yaw_keep_abs,
                        "mode_feature_version": getattr(model, "mode_feature_version", "unknown"),
                        "split_info": split_info,
                        "training_schedule": {
                            "stage_m_epochs": args.stage_m_epochs,
                            "stage_r_epochs": args.stage_r_epochs,
                            "stage_j_epochs": stage_j_epochs,
                        },
                        "history": history,
                        "model_kind": "student_candidate_evaluator_v2",
                        "selected_epoch": int(epoch),
                        "selection_metric": "predicted_yaw_needed_candidate_regret_mean",
                    },
                    output_dir / "student_candidate_evaluator_v2_best_rank.pt",
                )
        if args.candidate_scope == "yaw_aware":
            score = (
                predicted_metrics.get("yaw_needed_candidate_regret_mean", predicted_metrics["loss"]) * 1.0
                + predicted_metrics.get("candidate_regret_mean", predicted_metrics["loss"]) * 0.2
                - predicted_metrics.get("mode_balanced_acc", 0.0) * 2.0
                - predicted_metrics.get("yaw_needed_pairwise_accuracy", 0.0) * 0.5
            )
        else:
            score = predicted_metrics.get("candidate_regret_mean", predicted_metrics["loss"])
        if score < best:
            best = score
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "dataset_npz": args.dataset_npz,
                    "handoff_state_ckpt": args.handoff_state_ckpt,
                    "candidate_scope": args.candidate_scope,
                    "yaw_mode_enabled": True,
                    "yaw_mode_num_classes": int(yaw_mode_num_classes),
                    "yaw_keep_abs": args.yaw_keep_abs,
                    "mode_feature_version": getattr(model, "mode_feature_version", "unknown"),
                    "split_info": split_info,
                    "training_schedule": {
                        "stage_m_epochs": args.stage_m_epochs,
                        "stage_r_epochs": args.stage_r_epochs,
                        "stage_j_epochs": stage_j_epochs,
                    },
                    "history": history,
                    "model_kind": "student_candidate_evaluator_v2",
                },
                output_dir / "student_candidate_evaluator_v2_best.pt",
            )
    (output_dir / "train_history.json").write_text(json.dumps(history, indent=2))
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "dataset_npz": args.dataset_npz,
            "handoff_state_ckpt": args.handoff_state_ckpt,
            "candidate_scope": args.candidate_scope,
            "yaw_mode_enabled": True,
            "yaw_mode_num_classes": int(yaw_mode_num_classes),
            "yaw_keep_abs": args.yaw_keep_abs,
            "mode_feature_version": getattr(model, "mode_feature_version", "unknown"),
            "split_info": split_info,
            "training_schedule": {
                "stage_m_epochs": args.stage_m_epochs,
                "stage_r_epochs": args.stage_r_epochs,
                "stage_j_epochs": stage_j_epochs,
            },
            "history": history,
            "model_kind": "student_candidate_evaluator_v2",
        },
        output_dir / "student_candidate_evaluator_v2_final.pt",
    )
    mode_report, candidate_report, decision = _build_gate_reports(history)
    if best_mode_epoch >= 0:
        mode_report["saved_checkpoint"] = "student_candidate_evaluator_v2_best_mode.pt"
        mode_report["saved_checkpoint_epoch"] = int(best_mode_epoch)
    if best_rank_epoch >= 0:
        candidate_report["saved_checkpoint"] = "student_candidate_evaluator_v2_best_rank.pt"
        candidate_report["saved_checkpoint_epoch"] = int(best_rank_epoch)
    (output_dir / "mode_gate_report.json").write_text(json.dumps(mode_report, indent=2))
    (output_dir / "candidate_regret_gate_report.json").write_text(json.dumps(candidate_report, indent=2))
    (output_dir / "offline_gate_decision.json").write_text(json.dumps(decision, indent=2))
    if best_rank_state is not None:
        model.load_state_dict(best_rank_state, strict=False)
    focus_eps = [int(x) for x in str(args.focus_episodes).split(",") if x.strip()]
    focus_diag = build_focus_episode_diagnostics(
        model,
        handoff_model,
        dataset,
        focus_eps,
        device,
        args.candidate_scope,
        args.yaw_keep_abs,
        args.batch_size,
    )
    (output_dir / "focus_episode_diagnostics.json").write_text(json.dumps(focus_diag, indent=2))


if __name__ == "__main__":
    main()
