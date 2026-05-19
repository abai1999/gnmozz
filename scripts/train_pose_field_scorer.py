"""
train_pose_field_scorer.py

Train a local candidate-action scorer for basin alignment.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import BatchSampler, DataLoader, Dataset, Subset, WeightedRandomSampler

from prismatic.models.pose_field_scorer import PoseFieldScorer


class PoseCandidateDataset(Dataset):
    def __init__(self, npz_path: str):
        self.npz_path = str(npz_path)
        data = np.load(npz_path)
        self.data = {k: data[k] for k in data.files}
        self.n = int(self.data["wrist_depth"].shape[0])
        self.score_key = "candidate_oracle_score" if "candidate_oracle_score" in self.data else "candidate_improvement"
        self.meta = {}
        meta_path = Path(npz_path).with_suffix(".meta.json")
        if meta_path.exists():
            try:
                self.meta = json.loads(meta_path.read_text())
            except Exception:
                self.meta = {}

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        proxy_delta_key = "proxy_current_delta_basin_target" if "proxy_current_delta_basin_target" in self.data else "current_delta_basin_target"
        teacher_delta = (
            self.data["teacher_current_delta_basin_target"][idx]
            if "teacher_current_delta_basin_target" in self.data
            else (
                self.data["target_delta_teacher"][idx]
                if "target_delta_teacher" in self.data
                else self.data[proxy_delta_key][idx]
            )
        )
        if "planner_close_intent" in self.data:
            planner_close_intent = float(self.data["planner_close_intent"][idx])
        else:
            planner_close_intent = float(self.data["gripper_context"][idx][1] <= 0.5)
        return {
            "front_rgb": torch.from_numpy(
                (
                    self.data["front_rgb"][idx]
                    if "front_rgb" in self.data
                    else np.zeros((128, 128, 3), dtype=np.uint8)
                ).astype(np.float32).transpose(2, 0, 1)
            ) / 255.0,
            "wrist_rgb": torch.from_numpy(
                (
                    self.data["wrist_rgb"][idx]
                    if "wrist_rgb" in self.data
                    else np.zeros((128, 128, 3), dtype=np.uint8)
                ).astype(np.float32).transpose(2, 0, 1)
            ) / 255.0,
            "wrist_depth": torch.from_numpy(self.data["wrist_depth"][idx]),
            "proprio": torch.from_numpy(self.data["proprio"][idx]),
            "base_action": torch.from_numpy(self.data["base_action"][idx]),
            "gripper_context": torch.from_numpy(self.data["gripper_context"][idx]),
            "step_idx": torch.tensor(int(self.data["step_idx"][idx]), dtype=torch.long),
            "phase_id": torch.tensor(int(self.data["phase_id"][idx]), dtype=torch.long),
            "phase_age": torch.tensor(float(self.data["phase_age"][idx]), dtype=torch.float32),
            "steps_since_last_replan": torch.tensor(float(self.data["steps_since_last_replan"][idx]), dtype=torch.float32),
            "candidate_actions_local": torch.from_numpy(self.data["candidate_actions_local"][idx]),
            "candidate_group_index": torch.from_numpy(self.data["candidate_group_index"][idx]),
            "candidate_mask": torch.from_numpy(
                self.data["candidate_mask"][idx] if "candidate_mask" in self.data else np.ones_like(self.data["candidate_group_index"][idx], dtype=np.float32)
            ),
            "candidate_improvement": torch.from_numpy(self.data["candidate_improvement"][idx]),
            "candidate_oracle_score": torch.from_numpy(self.data[self.score_key][idx]),
            "candidate_next_basin_distance": torch.from_numpy(self.data["candidate_next_basin_distance"][idx]),
            "candidate_tier": torch.from_numpy(self.data["candidate_tier"][idx]),
            "current_delta_basin_target": torch.from_numpy(self.data[proxy_delta_key][idx]),
            "teacher_current_delta_basin_target": torch.from_numpy(np.asarray(teacher_delta, dtype=np.float32)),
            "current_basin_distance": torch.tensor(float(self.data["current_basin_distance"][idx]), dtype=torch.float32),
            "planner_close_intent": torch.tensor(planner_close_intent, dtype=torch.float32),
            "ready_to_close_target": torch.tensor(
                float(self.data["ready_to_close_target"][idx]) if "ready_to_close_target" in self.data else 0.0,
                dtype=torch.float32,
            ),
            "basin_distance_bin": torch.tensor(int(self.data["basin_distance_bin"][idx]), dtype=torch.long),
            "current_dx_sign": torch.tensor(int(self.data["current_dx_sign"][idx]), dtype=torch.long),
            "current_dy_sign": torch.tensor(int(self.data["current_dy_sign"][idx]), dtype=torch.long),
            "current_dyaw_sign": torch.tensor(int(self.data["current_dyaw_sign"][idx]), dtype=torch.long),
            "best_candidate_index": torch.tensor(int(self.data["best_candidate_index"][idx]), dtype=torch.long),
            "best_group_index": torch.tensor(int(self.data["best_group_index"][idx]), dtype=torch.long),
            "sample_weight": torch.tensor(
                float(self.data["sample_weight"][idx]) if "sample_weight" in self.data else 1.0,
                dtype=torch.float32,
            ),
            "yaw_hard_negative": torch.tensor(
                float(self.data["yaw_hard_negative"][idx]) if "yaw_hard_negative" in self.data else 0.0,
                dtype=torch.float32,
            ),
            "yaw_hard_positive": torch.tensor(
                float(self.data["yaw_hard_positive"][idx]) if "yaw_hard_positive" in self.data else 0.0,
                dtype=torch.float32,
            ),
            "xy_focus": torch.tensor(
                float(self.data["xy_focus"][idx]) if "xy_focus" in self.data else 0.0,
                dtype=torch.float32,
            ),
            "source_domain": torch.tensor(
                int(self.data["source_domain"][idx]) if "source_domain" in self.data else 0,
                dtype=torch.long,
            ),
        }


def split_by_episode(dataset: PoseCandidateDataset, val_ratio: float, seed: int):
    if "episode_index" not in dataset.data:
        raise RuntimeError(
            "dataset is missing `episode_index`; episode-level split is required for pose-field training."
        )
    episode_index = np.asarray(dataset.data["episode_index"], dtype=np.int64)
    unique_episodes = np.unique(episode_index)
    if unique_episodes.size < 2:
        raise RuntimeError(
            f"need at least 2 unique episodes for episode-level split, got {unique_episodes.size}"
        )
    rng = np.random.default_rng(seed)
    rng.shuffle(unique_episodes)
    val_episode_count = max(1, int(round(unique_episodes.size * float(val_ratio))))
    if val_episode_count >= unique_episodes.size:
        val_episode_count = max(1, unique_episodes.size - 1)
    val_episodes = set(unique_episodes[:val_episode_count].tolist())
    train_indices = [i for i, ep in enumerate(episode_index.tolist()) if ep not in val_episodes]
    val_indices = [i for i, ep in enumerate(episode_index.tolist()) if ep in val_episodes]
    if not train_indices or not val_indices:
        raise RuntimeError(
            f"episode-level split produced empty subset: train={len(train_indices)} val={len(val_indices)}"
        )
    return train_indices, val_indices


def load_pose_field_init_checkpoint(ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    dataset_npz = ckpt.get("dataset_npz")
    if dataset_npz is None:
        raise ValueError(f"Pose-field init checkpoint missing dataset_npz: {ckpt_path}")
    dataset = np.load(dataset_npz)
    candidate_group_index = dataset["candidate_group_index"][0].astype(np.int64)
    num_candidate_groups = int(ckpt.get("num_candidate_groups", int(np.max(candidate_group_index)) + 1))
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
    state_dict = ckpt["model_state_dict"]
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print(f"[pose_field] init checkpoint compatibility: missing={missing}, unexpected={unexpected}")
    return model, ckpt


def maybe_freeze_backbone(model: PoseFieldScorer, freeze_visual: bool, freeze_state_mlp: bool):
    if freeze_visual:
        for module in [
            model.depth_encoder,
            model.front_rgb_encoder,
            model.wrist_rgb_encoder,
            model.proprio_encoder,
            model.base_action_encoder,
            model.gripper_encoder,
            model.step_embedding,
            model.phase_embedding,
            model.delta_encoder,
            model.dx_sign_embedding,
            model.dy_sign_embedding,
            model.dyaw_sign_embedding,
            model.basin_bin_embedding,
            model.context_encoder,
        ]:
            for p in module.parameters():
                p.requires_grad = False
    if freeze_state_mlp:
        for p in model.state_mlp.parameters():
            p.requires_grad = False


def maybe_freeze_heads(
    model: PoseFieldScorer,
    freeze_group_head: bool,
    freeze_step_scale_head: bool,
    freeze_ready_head: bool,
):
    if freeze_group_head:
        for p in model.group_head.parameters():
            p.requires_grad = False
    if freeze_step_scale_head:
        for p in model.step_scale_head.parameters():
            p.requires_grad = False
    if freeze_ready_head:
        for p in model.ready_head.parameters():
            p.requires_grad = False


def weighted_mean(loss: torch.Tensor, sample_weight: torch.Tensor) -> torch.Tensor:
    sw = sample_weight.reshape(-1).to(device=loss.device, dtype=loss.dtype)
    vals = loss.reshape(-1)
    return (vals * sw).sum() / torch.clamp(sw.sum(), min=1e-6)


def masked_group_argmax(
    scores: torch.Tensor,
    candidate_group_index: torch.Tensor,
    group_index: torch.Tensor,
    candidate_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    mask = candidate_group_index.eq(group_index.unsqueeze(1))
    if candidate_mask is not None:
        mask = mask & (candidate_mask > 0.5)
    masked_scores = scores.masked_fill(~mask, -1e9)
    return masked_scores.argmax(dim=-1)


def compute_step_scale_target(
    current_delta: torch.Tensor,
    candidate_actions: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    """Heuristic local line-search target for the per-candidate step controller.

    It treats candidate_actions as local deltas and chooses the scale in a small
    discrete grid that most reduces the normalized target delta. This teaches
    the model when to take full 4-8mm primitives and when to shrink near the
    handoff band, without turning the scorer into a task FSM.
    """
    scales = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0], device=current_delta.device, dtype=current_delta.dtype)
    weights = torch.tensor([1.0 / 0.008, 1.0 / 0.008, 1.0 / 0.010, 0.0, 0.0, 1.0 / 0.050], device=current_delta.device, dtype=current_delta.dtype)
    residual = current_delta[:, None, None, :] - scales.view(1, 1, -1, 1) * candidate_actions[:, :, None, :]
    cost = torch.linalg.norm(residual * weights.view(1, 1, 1, 6), dim=-1)
    best_scale = scales[cost.argmin(dim=-1)]
    cand_norm = torch.linalg.norm(candidate_actions, dim=-1)
    best_scale = torch.where((cand_norm > 1e-8) & valid_mask, best_scale, torch.zeros_like(best_scale))
    return best_scale.detach()


def evaluate(model, loader, device, target_temperature, global_candidate_ranking: bool = True):
    model.eval()
    total = 0
    top1 = 0
    group_top1 = 0
    selected_improve = []
    selected_oracle_score = []
    oracle_best_score = []
    selected_dist = []
    pred_hist = []
    pred_group_hist = []
    all_pred = []
    all_best = []
    all_pred_prob = []
    all_logit_margin = []
    all_planner_close_intent = []
    all_z_abs = []
    all_ready_target = []
    all_ready_prob = []
    all_ready_pred = []
    with torch.no_grad():
        for batch in loader:
            fr = batch["front_rgb"].to(device=device, dtype=torch.float32)
            wr = batch["wrist_rgb"].to(device=device, dtype=torch.float32)
            wd = batch["wrist_depth"].to(device=device, dtype=torch.float32)
            pr = batch["proprio"].to(device=device, dtype=torch.float32)
            ba = batch["base_action"].to(device=device, dtype=torch.float32)
            gc = batch["gripper_context"].to(device=device, dtype=torch.float32)
            si = batch["step_idx"].to(device=device)
            pid = batch["phase_id"].to(device=device)
            page = batch["phase_age"].to(device=device, dtype=torch.float32)
            sr = batch["steps_since_last_replan"].to(device=device, dtype=torch.float32)
            ca = batch["candidate_actions_local"].to(device=device, dtype=torch.float32)
            cgi = batch["candidate_group_index"].to(device=device)
            cmask = batch["candidate_mask"].to(device=device, dtype=torch.float32)
            best = batch["best_candidate_index"].to(device=device)
            best_group = batch["best_group_index"].to(device=device)
            improve = batch["candidate_improvement"].to(device=device, dtype=torch.float32)
            oracle_score = batch["candidate_oracle_score"].to(device=device, dtype=torch.float32)
            oracle_valid = oracle_score > -1e8
            next_dist = batch["candidate_next_basin_distance"].to(device=device, dtype=torch.float32)
            cur_delta_metrics = batch["current_delta_basin_target"].to(device=device, dtype=torch.float32)
            use_target_context = bool(getattr(model, "use_target_context", True))
            cur_delta = batch["current_delta_basin_target"].to(device=device, dtype=torch.float32) if use_target_context else None
            dxs = batch["current_dx_sign"].to(device=device) if use_target_context else None
            dys = batch["current_dy_sign"].to(device=device) if use_target_context else None
            dyaws = batch["current_dyaw_sign"].to(device=device) if use_target_context else None
            basin_bin = batch["basin_distance_bin"].to(device=device) if use_target_context else None
            planner_close_intent = batch["planner_close_intent"].to(device=device, dtype=torch.float32)
            ready_target = batch["ready_to_close_target"].to(device=device, dtype=torch.float32)
            outputs = model(
                fr, wr, wd, pr, ba, gc, si, ca,
                phase_id=pid, phase_age=page, steps_since_last_replan=sr,
                current_delta_basin_target=cur_delta,
                current_dx_sign=dxs,
                current_dy_sign=dys,
                current_dyaw_sign=dyaws,
                basin_distance_bin=basin_bin,
                candidate_mask=cmask,
                return_aux=True,
            )
            scores = outputs["candidate_scores"]
            step_scale = outputs.get("candidate_step_scale", torch.ones_like(scores))
            group_logits = outputs["group_logits"]
            ready_prob = outputs["ready_to_close"]
            valid_candidate_mask = (cmask > 0.5) & oracle_valid
            masked_oracle_all = oracle_score.masked_fill(~valid_candidate_mask, -1e9)
            oracle_best = masked_oracle_all.argmax(dim=-1)
            oracle_best_group = cgi.gather(1, oracle_best.unsqueeze(1)).squeeze(1)
            if global_candidate_ranking:
                masked_scores = scores.masked_fill(~valid_candidate_mask, -1e9)
                pred = masked_scores.argmax(dim=-1)
                pred_group = cgi.gather(1, pred.unsqueeze(1)).squeeze(1)
                pred_probs_all = torch.softmax(masked_scores, dim=-1)
                pred_prob = pred_probs_all.gather(1, pred.unsqueeze(1)).squeeze(1)
                group_margin = torch.zeros_like(pred_prob)
            else:
                group_valid = []
                for group_id in range(group_logits.shape[1]):
                    group_valid.append(torch.any((cgi == group_id) & valid_candidate_mask, dim=1))
                group_valid = torch.stack(group_valid, dim=1)
                pred_group = group_logits.masked_fill(~group_valid, -1e9).argmax(dim=-1)
                pred = masked_group_argmax(scores, cgi, pred_group, valid_candidate_mask.float())
                group_probs = torch.softmax(group_logits, dim=-1)
                selected_group_prob = group_probs.gather(1, pred_group.unsqueeze(1)).squeeze(1)
                group_mask = cgi.eq(pred_group.unsqueeze(1)) & valid_candidate_mask
                masked_scores = scores.masked_fill(~group_mask, -1e9)
                within_group_probs = torch.softmax(masked_scores, dim=-1)
                pred_prob = selected_group_prob * within_group_probs.gather(1, pred.unsqueeze(1)).squeeze(1)
                group_top2 = torch.topk(group_logits, k=min(2, group_logits.shape[1]), dim=-1).values
                group_margin = group_top2[:, 0] - group_top2[:, 1] if group_top2.shape[1] > 1 else torch.zeros_like(pred_prob)
            cand_top2 = torch.topk(masked_scores, k=min(2, masked_scores.shape[1]), dim=-1).values
            cand_margin = cand_top2[:, 0] - cand_top2[:, 1] if cand_top2.shape[1] > 1 else torch.zeros_like(pred_prob)
            logit_margin = group_margin + cand_margin
            top1 += int((pred == oracle_best).sum().item())
            group_top1 += int((pred_group == oracle_best_group).sum().item())
            total += int(oracle_best.shape[0])
            selected_improve.append(improve.gather(1, pred.unsqueeze(1)).squeeze(1).cpu().numpy())
            selected_oracle_score.append(oracle_score.gather(1, pred.unsqueeze(1)).squeeze(1).cpu().numpy())
            oracle_best_score.append(oracle_score.gather(1, oracle_best.unsqueeze(1)).squeeze(1).cpu().numpy())
            selected_dist.append(next_dist.gather(1, pred.unsqueeze(1)).squeeze(1).cpu().numpy())
            pred_hist.append(pred.cpu().numpy())
            pred_group_hist.append(pred_group.cpu().numpy())
            all_pred.append(pred.cpu().numpy())
            all_best.append(oracle_best.cpu().numpy())
            all_pred_prob.append(pred_prob.cpu().numpy())
            all_logit_margin.append(logit_margin.cpu().numpy())
            all_planner_close_intent.append(planner_close_intent.cpu().numpy())
            all_z_abs.append(torch.abs(cur_delta_metrics[:, 2]).cpu().numpy())
            all_ready_target.append(ready_target.cpu().numpy())
            all_ready_prob.append(ready_prob.cpu().numpy())
            all_ready_pred.append((ready_prob >= 0.5).float().cpu().numpy())
    pred_hist_arr = np.concatenate(pred_hist) if pred_hist else np.zeros((0,), dtype=np.int64)
    pred_group_hist_arr = np.concatenate(pred_group_hist) if pred_group_hist else np.zeros((0,), dtype=np.int64)
    pred_arr = np.concatenate(all_pred) if all_pred else np.zeros((0,), dtype=np.int64)
    best_arr = np.concatenate(all_best) if all_best else np.zeros((0,), dtype=np.int64)
    pred_prob_arr = np.concatenate(all_pred_prob) if all_pred_prob else np.zeros((0,), dtype=np.float32)
    logit_margin_arr = np.concatenate(all_logit_margin) if all_logit_margin else np.zeros((0,), dtype=np.float32)
    close_intent_arr = np.concatenate(all_planner_close_intent) if all_planner_close_intent else np.zeros((0,), dtype=np.float32)
    z_abs_arr = np.concatenate(all_z_abs) if all_z_abs else np.zeros((0,), dtype=np.float32)
    ready_target_arr = np.concatenate(all_ready_target) if all_ready_target else np.zeros((0,), dtype=np.float32)
    ready_prob_arr = np.concatenate(all_ready_prob) if all_ready_prob else np.zeros((0,), dtype=np.float32)
    ready_pred_arr = np.concatenate(all_ready_pred) if all_ready_pred else np.zeros((0,), dtype=np.float32)

    def split_metrics(mask):
        mask = np.asarray(mask, dtype=bool)
        if not np.any(mask):
            return {
                "count": 0,
                "top1_acc": 0.0,
                "candidate0_rate": 0.0,
                "nonzero_rate": 0.0,
                "top1_prob_mean": 0.0,
                "logit_margin_mean": 0.0,
                "pred_hist": {},
            }
        vals = pred_arr[mask]
        return {
            "count": int(mask.sum()),
            "top1_acc": float(np.mean(vals == best_arr[mask])),
            "candidate0_rate": float(np.mean(vals == 0)),
            "nonzero_rate": float(np.mean(vals != 0)),
            "top1_prob_mean": float(np.mean(pred_prob_arr[mask])),
            "logit_margin_mean": float(np.mean(logit_margin_arr[mask])),
            "pred_hist": {int(k): int(v) for k, v in zip(*np.unique(vals, return_counts=True))},
        }

    high_z_no_close = (close_intent_arr <= 0.5) & (z_abs_arr >= 0.025)
    low_z_or_close = (close_intent_arr > 0.5) | (z_abs_arr < 0.025)
    mid_z = (close_intent_arr <= 0.5) & (z_abs_arr >= 0.01) & (z_abs_arr < 0.025)
    ready_tp = float(np.sum((ready_pred_arr > 0.5) & (ready_target_arr > 0.5)))
    ready_fp = float(np.sum((ready_pred_arr > 0.5) & (ready_target_arr <= 0.5)))
    ready_fn = float(np.sum((ready_pred_arr <= 0.5) & (ready_target_arr > 0.5)))
    ready_precision = ready_tp / max(ready_tp + ready_fp, 1.0)
    ready_recall = ready_tp / max(ready_tp + ready_fn, 1.0)
    ready_f1 = 0.0 if (ready_precision + ready_recall) <= 1e-8 else (2.0 * ready_precision * ready_recall / (ready_precision + ready_recall))
    return {
        "top1_acc": float(top1 / max(total, 1)),
        "group_acc": float(group_top1 / max(total, 1)),
        "selected_improvement_mean": float(np.mean(np.concatenate(selected_improve))) if selected_improve else 0.0,
        "selected_oracle_score_mean": float(np.mean(np.concatenate(selected_oracle_score))) if selected_oracle_score else 0.0,
        "oracle_score_mean": float(np.mean(np.concatenate(oracle_best_score))) if oracle_best_score else 0.0,
        "oracle_regret_mean": float(np.mean(np.concatenate(oracle_best_score) - np.concatenate(selected_oracle_score))) if selected_oracle_score and oracle_best_score else 0.0,
        "oracle_regret_p95": float(np.percentile(np.concatenate(oracle_best_score) - np.concatenate(selected_oracle_score), 95)) if selected_oracle_score and oracle_best_score else 0.0,
        "selected_next_basin_distance_mean": float(np.mean(np.concatenate(selected_dist))) if selected_dist else 0.0,
        "pred_hist": {int(k): int(v) for k, v in zip(*np.unique(pred_hist_arr, return_counts=True))} if pred_hist else {},
        "pred_group_hist": {int(k): int(v) for k, v in zip(*np.unique(pred_group_hist_arr, return_counts=True))} if pred_group_hist else {},
        "candidate0_rate": float(np.mean(pred_arr == 0)) if pred_arr.size else 0.0,
        "nonzero_correction_rate": float(np.mean(pred_arr != 0)) if pred_arr.size else 0.0,
        "top1_prob_mean": float(np.mean(pred_prob_arr)) if pred_prob_arr.size else 0.0,
        "logit_margin_mean": float(np.mean(logit_margin_arr)) if logit_margin_arr.size else 0.0,
        "ready_acc": float(np.mean((ready_pred_arr > 0.5) == (ready_target_arr > 0.5))) if ready_pred_arr.size else 0.0,
        "ready_precision": float(ready_precision),
        "ready_recall": float(ready_recall),
        "ready_f1": float(ready_f1),
        "ready_positive_rate": float(np.mean(ready_pred_arr > 0.5)) if ready_pred_arr.size else 0.0,
        "ready_target_positive_rate": float(np.mean(ready_target_arr > 0.5)) if ready_target_arr.size else 0.0,
        "ready_prob_mean": float(np.mean(ready_prob_arr)) if ready_prob_arr.size else 0.0,
        "split_metrics": {
            "high_z_no_close": split_metrics(high_z_no_close),
            "mid_z_no_close": split_metrics(mid_z),
            "low_z_or_close": split_metrics(low_z_or_close),
        },
    }


def build_train_sampler(dataset: PoseCandidateDataset, subset: Subset):
    subset_indices = np.asarray(subset.indices, dtype=np.int64)
    best_idx = dataset.data["best_group_index"][subset_indices].astype(np.int64)
    uniq, counts = np.unique(best_idx, return_counts=True)
    inv = {int(k): 1.0 / float(v) for k, v in zip(uniq, counts)}
    sample_weights = np.asarray([inv[int(v)] for v in best_idx], dtype=np.float64)
    if "sample_weight" in dataset.data:
        sample_weights = sample_weights * np.asarray(dataset.data["sample_weight"][subset_indices], dtype=np.float64)
    sample_weights = sample_weights / max(sample_weights.mean(), 1e-12)
    return WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True,
    ), {int(k): int(v) for k, v in zip(uniq, counts)}


class DepthStratifiedBatchSampler(BatchSampler):
    def __init__(
        self,
        dataset: PoseCandidateDataset,
        subset: Subset,
        batch_size: int,
        *,
        seed: int = 3407,
        high_z_threshold: float = 0.025,
        low_z_threshold: float = 0.025,
        high_fraction: float = 0.4,
        mid_fraction: float = 0.2,
        low_fraction: float = 0.35,
        ready_fraction: float = 0.05,
        yaw_fraction: float = 0.15,
        xy_fraction: float = 0.15,
    ):
        self.subset_indices = np.asarray(subset.indices, dtype=np.int64)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self._epoch = 0
        cur_delta_key = "proxy_current_delta_basin_target" if "proxy_current_delta_basin_target" in dataset.data else "current_delta_basin_target"
        cur_delta = dataset.data[cur_delta_key][self.subset_indices]
        z_abs = np.abs(cur_delta[:, 2]).astype(np.float32)
        close_intent = dataset.data["planner_close_intent"][self.subset_indices].astype(np.float32)
        ready = (
            dataset.data["ready_to_close_target"][self.subset_indices].astype(np.float32)
            if "ready_to_close_target" in dataset.data
            else np.zeros_like(z_abs, dtype=np.float32)
        )
        yaw_hard_negative = (
            dataset.data["yaw_hard_negative"][self.subset_indices].astype(np.float32)
            if "yaw_hard_negative" in dataset.data
            else np.zeros_like(z_abs, dtype=np.float32)
        )
        yaw_hard_positive = (
            dataset.data["yaw_hard_positive"][self.subset_indices].astype(np.float32)
            if "yaw_hard_positive" in dataset.data
            else np.zeros_like(z_abs, dtype=np.float32)
        )
        xy_focus = (
            dataset.data["xy_focus"][self.subset_indices].astype(np.float32)
            if "xy_focus" in dataset.data
            else np.zeros_like(z_abs, dtype=np.float32)
        )
        self.sample_weight = (
            dataset.data["sample_weight"][self.subset_indices].astype(np.float64)
            if "sample_weight" in dataset.data
            else np.ones_like(z_abs, dtype=np.float64)
        )
        local_positions = np.arange(self.subset_indices.size, dtype=np.int64)
        ready_mask = ready > 0.5
        yaw_mask = (yaw_hard_negative > 0.5) | (yaw_hard_positive > 0.5)
        xy_mask = xy_focus > 0.5
        high_mask = (close_intent <= 0.5) & (z_abs >= float(high_z_threshold)) & ~ready_mask & ~yaw_mask & ~xy_mask
        low_mask = ((close_intent > 0.5) | (z_abs < float(low_z_threshold))) & ~ready_mask & ~yaw_mask & ~xy_mask
        mid_mask = ~(high_mask | low_mask | ready_mask | yaw_mask | xy_mask)
        self.groups = {
            "high_z_no_close": local_positions[high_mask],
            "mid_z_no_close": local_positions[mid_mask],
            "low_z_or_close": local_positions[low_mask],
            "ready": local_positions[ready_mask],
            "yaw_focus": local_positions[yaw_mask],
            "xy_focus": local_positions[xy_mask],
        }
        self.group_counts = {k: int(v.size) for k, v in self.groups.items()}
        self.group_weights = {
            "high_z_no_close": float(high_fraction),
            "mid_z_no_close": float(mid_fraction),
            "low_z_or_close": float(low_fraction),
            "ready": float(ready_fraction),
            "yaw_focus": float(yaw_fraction),
            "xy_focus": float(xy_fraction),
        }
        self.num_batches = max(1, int(np.ceil(len(self.subset_indices) / max(self.batch_size, 1))))

    def __len__(self):
        return self.num_batches

    def _batch_counts(self):
        active = [k for k, v in self.groups.items() if v.size > 0]
        if not active:
            return {}
        total_weight = sum(self.group_weights[k] for k in active)
        raw = {k: self.group_weights[k] / max(total_weight, 1e-8) * self.batch_size for k in active}
        counts = {k: int(np.floor(v)) for k, v in raw.items()}
        remainder = self.batch_size - sum(counts.values())
        if remainder > 0:
            frac = sorted(((raw[k] - counts[k], k) for k in active), reverse=True)
            for _, key in frac[:remainder]:
                counts[key] += 1
        return counts

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self._epoch)
        self._epoch += 1
        counts = self._batch_counts()
        for _ in range(self.num_batches):
            batch = []
            for key, count in counts.items():
                if count <= 0:
                    continue
                src = self.groups[key]
                if src.size <= 0:
                    continue
                probs = self.sample_weight[src].astype(np.float64)
                probs = probs / max(probs.sum(), 1e-12)
                picked = rng.choice(src, size=count, replace=True, p=probs)
                batch.extend(int(x) for x in picked.tolist())
            while len(batch) < self.batch_size and self.subset_indices.size > 0:
                batch.append(int(rng.choice(self.subset_indices)))
            rng.shuffle(batch)
            yield batch[: self.batch_size]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_npz", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--target_temperature", type=float, default=0.5)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--episode_level_split", action="store_true", default=True)
    parser.add_argument("--row_level_split", dest="episode_level_split", action="store_false")
    parser.add_argument("--no_depth", action="store_true", default=False)
    parser.add_argument("--no_base_action", action="store_true", default=False)
    parser.add_argument("--no_proprio", action="store_true", default=False)
    parser.add_argument("--no_target_context", action="store_true", default=False)
    parser.add_argument("--no_front_rgb", action="store_true", default=False)
    parser.add_argument("--no_wrist_rgb", action="store_true", default=False)
    parser.add_argument("--lambda_soft", type=float, default=0.5)
    parser.add_argument("--lambda_group", type=float, default=1.0)
    parser.add_argument("--lambda_ce", type=float, default=1.0)
    parser.add_argument("--lambda_rank", type=float, default=1.0)
    parser.add_argument("--lambda_tier", type=float, default=0.5)
    parser.add_argument("--lambda_ready", type=float, default=0.25)
    parser.add_argument("--lambda_score_reg", type=float, default=0.25)
    parser.add_argument("--lambda_step_scale", type=float, default=0.35)
    parser.add_argument("--global_candidate_ranking", action="store_true", default=False)
    parser.add_argument("--two_stage_group_ranking", dest="global_candidate_ranking", action="store_false")
    parser.add_argument("--rank_margin", type=float, default=0.1)
    parser.add_argument("--use_balanced_sampler", action="store_true", default=True)
    parser.add_argument("--no_balanced_sampler", dest="use_balanced_sampler", action="store_false")
    parser.add_argument("--use_depth_stratified_sampler", action="store_true", default=True)
    parser.add_argument("--no_depth_stratified_sampler", dest="use_depth_stratified_sampler", action="store_false")
    parser.add_argument("--high_z_threshold", type=float, default=0.025)
    parser.add_argument("--low_z_threshold", type=float, default=0.025)
    parser.add_argument("--stratified_high_fraction", type=float, default=0.35)
    parser.add_argument("--stratified_mid_fraction", type=float, default=0.25)
    parser.add_argument("--stratified_low_fraction", type=float, default=0.30)
    parser.add_argument("--stratified_ready_fraction", type=float, default=0.10)
    parser.add_argument("--stratified_yaw_fraction", type=float, default=0.20)
    parser.add_argument("--stratified_xy_fraction", type=float, default=0.20)
    parser.add_argument("--fire_tradeoff_pose_ratio", type=float, default=0.98)
    parser.add_argument("--init_ckpt", type=str, default=None)
    parser.add_argument("--consistency_teacher_ckpt", type=str, default=None)
    parser.add_argument("--freeze_visual_backbone", action="store_true", default=False)
    parser.add_argument("--freeze_state_mlp", action="store_true", default=False)
    parser.add_argument("--freeze_group_head", action="store_true", default=False)
    parser.add_argument("--freeze_step_scale_head", action="store_true", default=False)
    parser.add_argument("--freeze_ready_head", action="store_true", default=False)
    parser.add_argument("--lambda_consistency_score", type=float, default=0.25)
    parser.add_argument("--lambda_consistency_group", type=float, default=0.10)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = PoseCandidateDataset(args.dataset_npz)
    if dataset.meta:
        print("[pose_field] dataset_meta:")
        print(json.dumps({
            "oracle_mode": dataset.meta.get("oracle_mode"),
            "support_close_intent_mode": dataset.meta.get("support_close_intent_mode"),
            "support_source_summary": dataset.meta.get("support_source_summary"),
            "support_high_z_no_close_summary": dataset.meta.get("support_high_z_no_close_summary"),
            "support_low_z_or_close_summary": dataset.meta.get("support_low_z_or_close_summary"),
            "yaw_focus": dataset.meta.get("yaw_focus"),
            "xy_focus": dataset.meta.get("xy_focus"),
            "oracle_thresholds": dataset.meta.get("oracle_thresholds"),
        }, indent=2))
    if args.episode_level_split:
        train_indices, val_indices = split_by_episode(dataset, args.val_ratio, args.seed)
        train_ds = Subset(dataset, train_indices)
        val_ds = Subset(dataset, val_indices)
    else:
        raise RuntimeError(
            "row-level random split is disabled for this training path; use episode-indexed datasets."
        )
    train_sampler = None
    train_batch_sampler = None
    train_hist = {}
    if args.use_depth_stratified_sampler:
        train_batch_sampler = DepthStratifiedBatchSampler(
            dataset,
            train_ds,
            args.batch_size,
            seed=args.seed,
            high_z_threshold=args.high_z_threshold,
            low_z_threshold=args.low_z_threshold,
            high_fraction=args.stratified_high_fraction,
            mid_fraction=args.stratified_mid_fraction,
            low_fraction=args.stratified_low_fraction,
            ready_fraction=args.stratified_ready_fraction,
            yaw_fraction=args.stratified_yaw_fraction,
            xy_fraction=args.stratified_xy_fraction,
        )
        train_hist = train_batch_sampler.group_counts
    elif args.use_balanced_sampler:
        train_sampler, train_hist = build_train_sampler(dataset, train_ds)
    if train_batch_sampler is not None:
        train_loader = DataLoader(train_ds, batch_sampler=train_batch_sampler, num_workers=0)
    else:
        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=(train_sampler is None),
            sampler=train_sampler,
            num_workers=0,
        )
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    num_candidate_groups = int(dataset.data["best_group_index"].max()) + 1
    ready_positive_rate = float(np.mean(dataset.data["ready_to_close_target"] > 0.5)) if "ready_to_close_target" in dataset.data else 0.0
    ready_pos = float(np.sum(dataset.data["ready_to_close_target"] > 0.5)) if "ready_to_close_target" in dataset.data else 0.0
    ready_neg = float(len(dataset) - ready_pos)
    ready_pos_weight = float(np.clip(ready_neg / max(ready_pos, 1.0), 1.0, 20.0))
    if args.init_ckpt:
        model, init_meta = load_pose_field_init_checkpoint(args.init_ckpt, device)
    else:
        model = PoseFieldScorer(
            use_depth=not args.no_depth,
            use_base_action=not args.no_base_action,
            use_proprio=not args.no_proprio,
            use_target_context=not args.no_target_context,
            use_front_rgb=not args.no_front_rgb,
            use_wrist_rgb=not args.no_wrist_rgb,
            num_candidate_groups=num_candidate_groups,
        ).to(device)
        init_meta = {}
    maybe_freeze_backbone(model, args.freeze_visual_backbone, args.freeze_state_mlp)
    maybe_freeze_heads(
        model,
        args.freeze_group_head,
        args.freeze_step_scale_head,
        args.freeze_ready_head,
    )
    teacher_model = None
    teacher_ckpt = args.consistency_teacher_ckpt or args.init_ckpt
    if teacher_ckpt:
        teacher_model, _ = load_pose_field_init_checkpoint(teacher_ckpt, device)
        teacher_model.eval()
        for p in teacher_model.parameters():
            p.requires_grad = False
    optim = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=args.weight_decay)

    best_val = -1e9
    best_state = None
    best_pose_epoch = -1
    epoch_snapshots = []
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        count = 0
        for batch in train_loader:
            fr = batch["front_rgb"].to(device=device, dtype=torch.float32)
            wr = batch["wrist_rgb"].to(device=device, dtype=torch.float32)
            wd = batch["wrist_depth"].to(device=device, dtype=torch.float32)
            pr = batch["proprio"].to(device=device, dtype=torch.float32)
            ba = batch["base_action"].to(device=device, dtype=torch.float32)
            gc = batch["gripper_context"].to(device=device, dtype=torch.float32)
            si = batch["step_idx"].to(device=device)
            pid = batch["phase_id"].to(device=device)
            page = batch["phase_age"].to(device=device, dtype=torch.float32)
            sr = batch["steps_since_last_replan"].to(device=device, dtype=torch.float32)
            ca = batch["candidate_actions_local"].to(device=device, dtype=torch.float32)
            cgi = batch["candidate_group_index"].to(device=device)
            cmask = batch["candidate_mask"].to(device=device, dtype=torch.float32)
            improve = batch["candidate_improvement"].to(device=device, dtype=torch.float32)
            oracle_score = batch["candidate_oracle_score"].to(device=device, dtype=torch.float32)
            oracle_valid = oracle_score > -1e8
            tier = batch["candidate_tier"].to(device=device)
            use_target_context = bool(getattr(model, "use_target_context", True))
            cur_delta = batch["current_delta_basin_target"].to(device=device, dtype=torch.float32) if use_target_context else None
            dxs = batch["current_dx_sign"].to(device=device) if use_target_context else None
            dys = batch["current_dy_sign"].to(device=device) if use_target_context else None
            dyaws = batch["current_dyaw_sign"].to(device=device) if use_target_context else None
            basin_bin = batch["basin_distance_bin"].to(device=device) if use_target_context else None
            best = batch["best_candidate_index"].to(device=device)
            best_group = batch["best_group_index"].to(device=device)
            ready_target = batch["ready_to_close_target"].to(device=device, dtype=torch.float32)
            sample_weight = batch["sample_weight"].to(device=device, dtype=torch.float32)
            source_domain = batch["source_domain"].to(device=device)
            outputs = model(
                fr, wr, wd, pr, ba, gc, si, ca,
                phase_id=pid, phase_age=page, steps_since_last_replan=sr,
                current_delta_basin_target=cur_delta,
                current_dx_sign=dxs,
                current_dy_sign=dys,
                current_dyaw_sign=dyaws,
                basin_distance_bin=basin_bin,
                candidate_mask=cmask,
                return_aux=True,
            )
            scores = outputs["candidate_scores"]
            step_scale = outputs.get("candidate_step_scale", torch.ones_like(scores))
            group_logits = outputs["group_logits"]
            ready_logits = outputs["ready_to_close_logits"]
            valid_candidate_mask = (cmask > 0.5) & oracle_valid
            masked_oracle_all = oracle_score.masked_fill(~valid_candidate_mask, -1e9)
            train_best = masked_oracle_all.argmax(dim=-1)
            train_best_group = cgi.gather(1, train_best.unsqueeze(1)).squeeze(1)
            group_mask = valid_candidate_mask if args.global_candidate_ranking else (cgi.eq(train_best_group.unsqueeze(1)) & valid_candidate_mask)
            masked_scores = scores.masked_fill(~group_mask, -1e9)
            masked_oracle = oracle_score.masked_fill(~group_mask, -1e9)
            target = torch.softmax(masked_oracle / max(float(args.target_temperature), 1e-6), dim=-1)
            loss_soft = weighted_mean(-(target * F.log_softmax(masked_scores, dim=-1)).sum(dim=-1), sample_weight)
            loss_group = weighted_mean(F.cross_entropy(group_logits, train_best_group, reduction="none"), sample_weight)
            loss_ce = weighted_mean(F.cross_entropy(masked_scores, train_best, reduction="none"), sample_weight)
            tier_target = (tier >= 2).float()
            tier_weights = torch.where(tier >= 3, torch.full_like(scores, 2.0), torch.ones_like(scores))
            tier_mask = group_mask.float()
            tier_logits = scores.masked_fill(~group_mask, 0.0)
            loss_tier = F.binary_cross_entropy_with_logits(tier_logits, tier_target, weight=tier_weights * tier_mask)
            valid_scores = scores.masked_fill(~group_mask, 0.0)
            valid_oracle = oracle_score.masked_fill(~group_mask, 0.0)
            denom = torch.clamp(group_mask.float().sum(dim=1, keepdim=True), min=1.0)
            score_mean = valid_scores.sum(dim=1, keepdim=True) / denom
            oracle_mean = valid_oracle.sum(dim=1, keepdim=True) / denom
            score_std = torch.sqrt(torch.clamp((((valid_scores - score_mean) ** 2) * group_mask.float()).sum(dim=1, keepdim=True) / denom, min=1e-6))
            oracle_std = torch.sqrt(torch.clamp((((valid_oracle - oracle_mean) ** 2) * group_mask.float()).sum(dim=1, keepdim=True) / denom, min=1e-6))
            score_z = (scores - score_mean) / score_std
            oracle_z = (oracle_score - oracle_mean) / oracle_std
            loss_score_reg = F.smooth_l1_loss(score_z[group_mask], oracle_z[group_mask]) if torch.any(group_mask) else scores.new_zeros(())
            scale_target = compute_step_scale_target(cur_delta if cur_delta is not None else torch.zeros_like(ba), ca, group_mask)
            loss_step_scale = F.smooth_l1_loss(step_scale[group_mask], scale_target[group_mask]) if torch.any(group_mask) else scores.new_zeros(())
            best_scores = scores.gather(1, train_best.unsqueeze(1))
            best_oracle_score = oracle_score.gather(1, train_best.unsqueeze(1))
            improve_gap = torch.clamp(best_oracle_score - oracle_score, min=0.0)
            rank_mask = (improve_gap > 1e-6) & group_mask
            rank_penalty = F.relu(args.rank_margin - (best_scores - scores))
            if torch.any(rank_mask):
                rank_per_sample = (rank_penalty * improve_gap * rank_mask.float()).sum(dim=1) / torch.clamp((improve_gap * rank_mask.float()).sum(dim=1), min=1e-6)
                loss_rank = weighted_mean(rank_per_sample, sample_weight)
            else:
                loss_rank = scores.new_zeros(())
            loss_ready = weighted_mean(F.binary_cross_entropy_with_logits(
                ready_logits,
                ready_target,
                reduction="none",
                pos_weight=torch.full((1,), ready_pos_weight, device=device, dtype=ready_logits.dtype),
            ), sample_weight)
            loss_consistency_score = scores.new_zeros(())
            loss_consistency_group = scores.new_zeros(())
            coarse_mask = source_domain == 0
            if teacher_model is not None and torch.any(coarse_mask):
                with torch.no_grad():
                    teacher_outputs = teacher_model(
                        fr, wr, wd, pr, ba, gc, si, ca,
                        phase_id=pid, phase_age=page, steps_since_last_replan=sr,
                        current_delta_basin_target=cur_delta,
                        current_dx_sign=dxs,
                        current_dy_sign=dys,
                        current_dyaw_sign=dyaws,
                        basin_distance_bin=basin_bin,
                        candidate_mask=cmask,
                        return_aux=True,
                    )
                teacher_scores = teacher_outputs["candidate_scores"]
                teacher_groups = teacher_outputs["group_logits"]
                coarse_weight = sample_weight[coarse_mask]
                student_logp = F.log_softmax(scores[coarse_mask], dim=-1)
                teacher_prob = F.softmax(teacher_scores[coarse_mask], dim=-1)
                kl_score = F.kl_div(student_logp, teacher_prob, reduction="none").sum(dim=-1)
                loss_consistency_score = weighted_mean(kl_score, coarse_weight)
                student_group_logp = F.log_softmax(group_logits[coarse_mask], dim=-1)
                teacher_group_prob = F.softmax(teacher_groups[coarse_mask], dim=-1)
                kl_group = F.kl_div(student_group_logp, teacher_group_prob, reduction="none").sum(dim=-1)
                loss_consistency_group = weighted_mean(kl_group, coarse_weight)
            loss = (
                args.lambda_soft * loss_soft
                + args.lambda_group * loss_group
                + args.lambda_ce * loss_ce
                + args.lambda_rank * loss_rank
                + args.lambda_tier * loss_tier
                + args.lambda_ready * loss_ready
                + args.lambda_score_reg * loss_score_reg
                + args.lambda_step_scale * loss_step_scale
                + args.lambda_consistency_score * loss_consistency_score
                + args.lambda_consistency_group * loss_consistency_group
            )
            optim.zero_grad()
            loss.backward()
            optim.step()
            running += float(loss.item()) * wd.shape[0]
            count += int(wd.shape[0])

        train_loss = running / max(count, 1)
        val_metrics = evaluate(model, val_loader, device, args.target_temperature, global_candidate_ranking=bool(args.global_candidate_ranking))
        history.append({"epoch": epoch, "train_loss": train_loss, **val_metrics})
        high_split = val_metrics["split_metrics"]["high_z_no_close"]
        low_split = val_metrics["split_metrics"]["low_z_or_close"]
        print(
            f"[pose_field] epoch={epoch} loss={train_loss:.4f} "
            f"val_group={val_metrics['group_acc']:.3f} "
            f"val_top1={val_metrics['top1_acc']:.3f} "
            f"val_sel_improve={val_metrics['selected_improvement_mean']:.4f} "
            f"val_sel_oracle={val_metrics['selected_oracle_score_mean']:.4f} "
            f"val_regret={val_metrics['oracle_regret_mean']:.4f} "
            f"val_regret_p95={val_metrics['oracle_regret_p95']:.4f} "
            f"val_c0={val_metrics['candidate0_rate']:.3f} "
            f"val_topprob={val_metrics['top1_prob_mean']:.3f} "
            f"val_ready_f1={val_metrics['ready_f1']:.3f} "
            f"val_ready_pos={val_metrics['ready_positive_rate']:.3f} "
            f"high_top1={high_split['top1_acc']:.3f} "
            f"high_prob={high_split['top1_prob_mean']:.3f} "
            f"low_top1={low_split['top1_acc']:.3f} "
            f"low_prob={low_split['top1_prob_mean']:.3f}"
        )
        pose_metric = -float(val_metrics["oracle_regret_mean"])
        if pose_metric > best_val:
            best_val = pose_metric
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            best_pose_epoch = int(epoch)
        epoch_snapshots.append(
            {
                "epoch": int(epoch),
                "selected_oracle_score_mean": float(val_metrics["selected_oracle_score_mean"]),
                "oracle_regret_mean": float(val_metrics["oracle_regret_mean"]),
                "ready_f1": float(val_metrics["ready_f1"]),
                "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            }
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_pose_state = best_state if best_state is not None else {k: v.detach().cpu() for k, v in model.state_dict().items()}
    best_regret = -float(best_val)
    regret_ceiling = best_regret / max(float(args.fire_tradeoff_pose_ratio), 1e-6)
    eligible_snapshots = [s for s in epoch_snapshots if float(s.get("oracle_regret_mean", np.inf)) <= regret_ceiling]
    if eligible_snapshots:
        best_fire_snapshot = max(
            eligible_snapshots,
            key=lambda s: (float(s["ready_f1"]), -float(s.get("oracle_regret_mean", np.inf)), -int(s["epoch"])),
        )
        best_fire_state = best_fire_snapshot["state_dict"]
        best_fire_tradeoff_epoch = int(best_fire_snapshot["epoch"])
        best_fire_tradeoff_ready_f1 = float(best_fire_snapshot["ready_f1"])
    else:
        best_fire_state = best_pose_state
        best_fire_tradeoff_epoch = int(best_pose_epoch)
        best_fire_tradeoff_ready_f1 = 0.0

    def _build_ckpt(state_dict, checkpoint_metric: str, *, selected_epoch: int, selected_ready_f1: float | None = None):
        return {
            "module_type": "pose_field_scorer",
            "model_state_dict": state_dict,
            "dataset_npz": args.dataset_npz,
            "dataset_meta": dataset.meta,
            "score_key": dataset.score_key,
            "use_depth": not args.no_depth,
            "use_base_action": not args.no_base_action,
            "use_proprio": not args.no_proprio,
            "use_target_context": not args.no_target_context,
            "use_front_rgb": not args.no_front_rgb,
            "use_wrist_rgb": not args.no_wrist_rgb,
            "fire_only_head": True,
            "num_candidate_groups": num_candidate_groups,
            "target_temperature": args.target_temperature,
            "train_best_group_hist": train_hist,
            "use_depth_stratified_sampler": bool(args.use_depth_stratified_sampler),
            "ready_positive_rate": ready_positive_rate,
            "ready_pos_weight": ready_pos_weight,
            "high_z_threshold": float(args.high_z_threshold),
            "low_z_threshold": float(args.low_z_threshold),
            "lambda_soft": args.lambda_soft,
            "lambda_group": args.lambda_group,
            "lambda_ce": args.lambda_ce,
            "lambda_rank": args.lambda_rank,
            "lambda_tier": args.lambda_tier,
            "lambda_ready": args.lambda_ready,
            "lambda_score_reg": args.lambda_score_reg,
            "lambda_step_scale": args.lambda_step_scale,
            "lambda_consistency_score": args.lambda_consistency_score,
            "lambda_consistency_group": args.lambda_consistency_group,
            "rank_margin": args.rank_margin,
            "selection_policy": "global_candidate" if args.global_candidate_ranking else "two_stage_group",
            "fire_tradeoff_pose_ratio": float(args.fire_tradeoff_pose_ratio),
            "init_ckpt": args.init_ckpt,
            "consistency_teacher_ckpt": teacher_ckpt,
            "freeze_visual_backbone": bool(args.freeze_visual_backbone),
            "freeze_state_mlp": bool(args.freeze_state_mlp),
            "freeze_group_head": bool(args.freeze_group_head),
            "freeze_step_scale_head": bool(args.freeze_step_scale_head),
            "freeze_ready_head": bool(args.freeze_ready_head),
            "best_val_oracle_regret_mean": float(-best_val),
            "best_pose_epoch": int(best_pose_epoch),
            "best_checkpoint_metric": checkpoint_metric,
            "selected_epoch": int(selected_epoch),
            "selected_ready_f1": None if selected_ready_f1 is None else float(selected_ready_f1),
        }
    best_pose_ckpt = _build_ckpt(best_pose_state, "oracle_regret_mean", selected_epoch=best_pose_epoch)
    best_fire_ckpt = _build_ckpt(
        best_fire_state,
        "ready_f1_under_pose_floor",
        selected_epoch=best_fire_tradeoff_epoch if best_fire_tradeoff_epoch >= 0 else best_pose_epoch,
        selected_ready_f1=best_fire_tradeoff_ready_f1 if best_fire_tradeoff_ready_f1 >= 0.0 else None,
    )
    torch.save(best_pose_ckpt, output_dir / "pose_field_scorer_best_pose.pt")
    torch.save(best_fire_ckpt, output_dir / "pose_field_scorer_best_fire_tradeoff.pt")
    torch.save(best_fire_ckpt, output_dir / "pose_field_scorer_final.pt")
    (output_dir / "history.json").write_text(json.dumps(history, indent=2))
    print(
        f"[pose_field] saved "
        f"{output_dir / 'pose_field_scorer_best_pose.pt'} and "
        f"{output_dir / 'pose_field_scorer_best_fire_tradeoff.pt'}"
    )


if __name__ == "__main__":
    main()
