#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler

from prismatic.models.student_handoff_state_head_v2 import StudentHandoffStateHeadV2


class AlignmentV3Dataset(Dataset):
    def __init__(self, npz_path: str):
        raw = np.load(npz_path, allow_pickle=False)
        self.data = {k: np.asarray(raw[k]) for k in raw.files}
        self.length = int(self.data["episode_index"].shape[0])

    def __len__(self):
        return self.length

    def __getitem__(self, idx: int) -> dict:
        front = torch.from_numpy(self.data["front_rgb"][idx].astype(np.float32).transpose(2, 0, 1)) / 255.0
        wrist = torch.from_numpy(self.data["wrist_rgb"][idx].astype(np.float32).transpose(2, 0, 1)) / 255.0
        depth = torch.from_numpy(self.data["wrist_depth"][idx].astype(np.float32))
        if depth.ndim == 2:
            depth = depth.unsqueeze(0)
        zeros = np.zeros((self.length,), dtype=np.float32)
        temporal_default = np.zeros((self.length, 32), dtype=np.float32)
        coarse5 = self.data.get("alignment_v3_corrective_dyaw_coarse_label", None)
        if coarse5 is None:
            sym = np.asarray(
                self.data.get(
                    "alignment_v3_corrective_dyaw_sym_label",
                    self.data.get("alignment_v3_corrective_dyaw_label", np.ones((self.length,), dtype=np.int64)),
                ),
                dtype=np.int64,
            )
            # Legacy fallback: keep the center bucket for zero, and map
            # negative/positive signs into the conservative coarse bins.
            coarse5 = np.where(sym == 0, 2, np.where(sym < 0, 1, 3))
        dyaw_residual = self.data.get("alignment_v3_corrective_dyaw_sym_residual", None)
        if dyaw_residual is None:
            dyaw_residual = np.zeros((self.length,), dtype=np.float32)
        return {
            "front_rgb": front,
            "wrist_rgb": wrist,
            "wrist_depth": depth,
            "proprio": torch.from_numpy(self.data["proprio"][idx].astype(np.float32)),
            "gripper_context": torch.from_numpy(self.data["gripper_context"][idx].astype(np.float32)),
            "proxy_current_delta_basin_target": torch.from_numpy(self.data["proxy_current_delta_basin_target"][idx].astype(np.float32)),
            "temporal_action_summary": torch.from_numpy(self.data.get("temporal_action_summary", temporal_default)[idx].astype(np.float32)),
            "current_dx_sign": torch.tensor(int(self.data["current_dx_sign"][idx]), dtype=torch.long),
            "current_dy_sign": torch.tensor(int(self.data["current_dy_sign"][idx]), dtype=torch.long),
            "current_dyaw_sign": torch.tensor(int(self.data["current_dyaw_sign"][idx]), dtype=torch.long),
            "basin_distance_bin": torch.tensor(int(self.data["basin_distance_bin"][idx]), dtype=torch.long),
            "substage_id": torch.tensor(int(self.data["substage_id"][idx]), dtype=torch.long),
            "contact_state": torch.tensor(int(self.data["contact_state"][idx]), dtype=torch.long),
            "stage_target_mode": torch.tensor(int(self.data["stage_target_mode"][idx]), dtype=torch.long),
            "teacher_metrics_norm": torch.from_numpy(self.data["teacher_metrics_norm"][idx].astype(np.float32)),
            "teacher_band_label": torch.tensor(int(self.data["teacher_band_label"][idx]), dtype=torch.long),
            "teacher_truth_handoff_ready": torch.tensor(float(self.data["teacher_truth_handoff_ready"][idx]), dtype=torch.float32),
            "sample_weight": torch.tensor(float(self.data["sample_weight"][idx]), dtype=torch.float32),
            "progress_label": torch.tensor(float(self.data.get("alignment_v2_progress_label", zeros)[idx]), dtype=torch.float32),
            "progress_mask": torch.tensor(float(self.data.get("alignment_v2_progress_mask", zeros)[idx]), dtype=torch.float32),
            "progress_ambiguous_mask": torch.tensor(float(self.data.get("alignment_v2_progress_ambiguous_mask", zeros)[idx]), dtype=torch.float32),
            "pair_prev_index": torch.tensor(int(self.data.get("alignment_v2_pair_prev_index", np.full((self.length,), -1, dtype=np.int64))[idx]), dtype=torch.long),
            "pair_label": torch.tensor(float(self.data.get("alignment_v2_pair_label", zeros)[idx]), dtype=torch.float32),
            "pair_mask": torch.tensor(float(self.data.get("alignment_v2_pair_mask", zeros)[idx]), dtype=torch.float32),
            "axis_block_label": torch.tensor(int(self.data.get("alignment_v2_axis_block_label", np.zeros((self.length,), dtype=np.int64))[idx]), dtype=torch.long),
            "closeability_label": torch.tensor(float(self.data.get("alignment_v3_closeability_label", zeros)[idx]), dtype=torch.float32),
            "closeability_borderline_mask": torch.tensor(float(self.data.get("alignment_v3_closeability_borderline_mask", zeros)[idx]), dtype=torch.float32),
            "corrective_dx_label": torch.tensor(int(self.data.get("alignment_v3_corrective_dx_label", np.ones((self.length,), dtype=np.int64))[idx]), dtype=torch.long),
            "corrective_dy_label": torch.tensor(int(self.data.get("alignment_v3_corrective_dy_label", np.ones((self.length,), dtype=np.int64))[idx]), dtype=torch.long),
            "corrective_dz_label": torch.tensor(int(self.data.get("alignment_v3_corrective_dz_label", np.ones((self.length,), dtype=np.int64))[idx]), dtype=torch.long),
            "corrective_dyaw_label": torch.tensor(int(self.data.get("alignment_v3_corrective_dyaw_label", np.ones((self.length,), dtype=np.int64))[idx]), dtype=torch.long),
            "corrective_dyaw_sym_label": torch.tensor(
                int(self.data.get("alignment_v3_corrective_dyaw_sym_label", self.data.get("alignment_v3_corrective_dyaw_label", np.ones((self.length,), dtype=np.int64)))[idx]),
                dtype=torch.long,
            ),
            "corrective_dyaw_coarse_label": torch.tensor(int(np.asarray(coarse5)[idx]), dtype=torch.long),
            "corrective_dyaw_sym_residual": torch.tensor(float(np.asarray(dyaw_residual)[idx]), dtype=torch.float32),
            "corrective_mask": torch.tensor(float(self.data.get("alignment_v3_corrective_mask", zeros)[idx]), dtype=torch.float32),
            "is_counterfactual": torch.tensor(float(self.data.get("is_counterfactual", zeros)[idx]), dtype=torch.float32),
            "far_negative_mask": torch.tensor(float(self.data.get("far_negative_mask_v2", zeros)[idx]), dtype=torch.float32),
            "boundary_mask": torch.tensor(float(self.data.get("boundary_band_mask_v2", zeros)[idx]), dtype=torch.float32),
            "source_name": str(self.data.get("source_name", np.full((self.length,), "unknown", dtype="U64"))[idx]),
        }


class PairwiseAlignmentV3Dataset(Dataset):
    def __init__(self, dataset: AlignmentV3Dataset, indices: list[int]):
        self.dataset = dataset
        self.indices = [int(i) for i in indices]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item: int) -> dict:
        cur_idx = self.indices[item]
        prev_idx = int(self.dataset.data["alignment_v2_pair_prev_index"][cur_idx])
        cur = self.dataset[cur_idx]
        prev = self.dataset[prev_idx]
        out: dict[str, object] = {}
        for key, value in cur.items():
            if key == "source_name":
                continue
            out[f"cur_{key}"] = value
        for key, value in prev.items():
            if key == "source_name":
                continue
            out[f"prev_{key}"] = value
        return out


def split_by_episode(dataset: AlignmentV3Dataset, val_ratio: float, seed: int):
    eps = np.asarray(dataset.data["episode_index"], dtype=np.int64)
    uniq = np.unique(eps)
    if uniq.size < 2:
        raise RuntimeError("need at least 2 episodes for validation split")
    rng = np.random.default_rng(seed)
    shuffled = uniq.copy()
    rng.shuffle(shuffled)
    val_n = max(1, int(round(uniq.size * val_ratio)))
    val_n = min(val_n, uniq.size - 1)
    val_eps = set(int(x) for x in shuffled[:val_n].tolist())
    if "alignment_v3_closeability_label" in dataset.data:
        closeability = np.asarray(dataset.data["alignment_v3_closeability_label"], dtype=np.float32) > 0.5
        positive_eps = sorted({int(e) for e in eps[closeability].tolist()})
        if positive_eps:
            # Keep validation honest: it needs at least one episode that
            # actually contains closeability-positive surrogate rows.
            if not any(ep in val_eps for ep in positive_eps):
                val_eps.add(int(positive_eps[0]))
                if len(val_eps) > val_n:
                    for ep in shuffled[::-1].tolist():
                        epi = int(ep)
                        if epi in val_eps and epi not in positive_eps:
                            val_eps.remove(epi)
                            break
    train_idx = [i for i, e in enumerate(eps.tolist()) if int(e) not in val_eps]
    val_idx = [i for i, e in enumerate(eps.tolist()) if int(e) in val_eps]
    return train_idx, val_idx, sorted(val_eps)


def cap_far_negative_val_rows(dataset: AlignmentV3Dataset, val_idx: list[int], cap_per_episode: int, seed: int) -> list[int]:
    if cap_per_episode <= 0:
        return val_idx
    eps = np.asarray(dataset.data["episode_index"], dtype=np.int64)
    source = np.asarray(dataset.data.get("source_name", np.full((dataset.length,), "unknown", dtype="U64"))).astype(str)
    rng = np.random.default_rng(seed)
    keep: list[int] = []
    val_by_ep: dict[int, list[int]] = {}
    for idx in val_idx:
        val_by_ep.setdefault(int(eps[idx]), []).append(int(idx))
    for ep, indices in sorted(val_by_ep.items()):
        far = [i for i in indices if source[i] == "far_negative"]
        other = [i for i in indices if source[i] != "far_negative"]
        if len(far) > cap_per_episode:
            far = rng.choice(np.asarray(far, dtype=np.int64), size=cap_per_episode, replace=False).astype(int).tolist()
        keep.extend(other)
        keep.extend(far)
    return sorted(keep)


def compute_binary_pos_weight(dataset: AlignmentV3Dataset, indices: list[int], key: str, smooth: float = 1.0, cap: float = 16.0) -> float:
    labels = np.asarray(dataset.data[key], dtype=np.float32)[indices] > 0.5
    pos = float(labels.sum())
    neg = float(labels.size - labels.sum())
    if pos <= 0.0:
        return 1.0
    weight = ((neg + smooth) / (pos + smooth)) ** 0.5
    return float(max(1.0, min(weight, cap)))


def compute_class_weights(
    dataset: AlignmentV3Dataset,
    indices: list[int],
    key: str,
    num_classes: int,
    smoothing: float = 1.0,
    power: float = 0.5,
    min_weight: float = 0.25,
    max_weight: float = 4.0,
) -> torch.Tensor:
    labels = np.asarray(dataset.data[key], dtype=np.int64)[indices]
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    total = float(counts.sum())
    weights = np.ones((num_classes,), dtype=np.float64)
    for cls in range(num_classes):
        weights[cls] = ((total + smoothing * num_classes) / (counts[cls] + smoothing)) ** float(power)
    weights = weights / max(float(weights.mean()), 1e-6)
    weights = np.clip(weights, min_weight, max_weight)
    return torch.as_tensor(weights, dtype=torch.float32)


def corrective_focus_mask(dataset: AlignmentV3Dataset, indices: list[int]) -> np.ndarray:
    closeability = np.asarray(dataset.data.get("alignment_v3_closeability_label", np.zeros((dataset.length,), dtype=np.float32)), dtype=np.float32) > 0.5
    borderline = np.asarray(
        dataset.data.get("alignment_v3_closeability_borderline_mask", np.zeros((dataset.length,), dtype=np.float32)),
        dtype=np.float32,
    ) > 0.5
    counterfactual = np.asarray(dataset.data.get("is_counterfactual", np.zeros((dataset.length,), dtype=np.float32)), dtype=np.float32) > 0.5
    mask = closeability | borderline | counterfactual
    return np.asarray([bool(mask[int(i)]) for i in indices], dtype=np.bool_)


def pair_indices_for_split(dataset: AlignmentV3Dataset, indices: list[int]) -> list[int]:
    if "alignment_v2_pair_prev_index" not in dataset.data or "alignment_v2_pair_mask" not in dataset.data:
        return []
    allowed = set(int(i) for i in indices)
    prev = np.asarray(dataset.data["alignment_v2_pair_prev_index"], dtype=np.int64)
    mask = np.asarray(dataset.data["alignment_v2_pair_mask"], dtype=np.float32) > 0.5
    return [int(i) for i in indices if bool(mask[int(i)]) and int(prev[int(i)]) in allowed and int(prev[int(i)]) >= 0]


def forward_model(model, batch, device):
    return model(
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
        temporal_action_summary=batch.get("temporal_action_summary", None).to(device=device, dtype=torch.float32)
        if batch.get("temporal_action_summary", None) is not None
        else None,
    )


def prefixed_batch(batch: dict, prefix: str) -> dict:
    plen = len(prefix)
    return {key[plen:]: value for key, value in batch.items() if key.startswith(prefix)}


def compute_loss(model, batch, device, args):
    out = forward_model(model, batch, device)
    pred_metrics = out["pred_metrics_norm"]
    band_logits = out["band_logits"]
    ready_logit = out["ready_logit"]
    progress_logit = out.get("progress_logit")
    axis_block_logits = out.get("axis_block_logits")
    closeability_logit = out.get("closeability_logit")
    corrective_dx_logits = out.get("corrective_dx_logits")
    corrective_dy_logits = out.get("corrective_dy_logits")
    corrective_dz_logits = out.get("corrective_dz_logits")
    corrective_dyaw_logits = out.get("corrective_dyaw_logits")
    corrective_dyaw_legacy_logits = out.get("corrective_dyaw_legacy_logits")
    corrective_dyaw_residual_pred = out.get("corrective_dyaw_residual")
    teacher_metrics = batch["teacher_metrics_norm"].to(device=device, dtype=torch.float32)
    teacher_band = batch["teacher_band_label"].to(device=device)
    teacher_ready = batch["teacher_truth_handoff_ready"].to(device=device, dtype=torch.float32)
    sample_weight = batch["sample_weight"].to(device=device, dtype=torch.float32)
    progress_label = batch["progress_label"].to(device=device, dtype=torch.float32)
    progress_mask = batch["progress_mask"].to(device=device, dtype=torch.float32)
    far_negative = batch["far_negative_mask"].to(device=device, dtype=torch.float32) > 0.5
    boundary_mask = batch["boundary_mask"].to(device=device, dtype=torch.float32) > 0.5
    axis_label = batch["axis_block_label"].to(device=device)
    closeability_label = batch["closeability_label"].to(device=device, dtype=torch.float32)
    closeability_borderline_mask = batch["closeability_borderline_mask"].to(device=device, dtype=torch.float32)
    corrective_mask = batch["corrective_mask"].to(device=device, dtype=torch.float32) > 0.5
    corrective_focus = (
        (closeability_label > 0.5)
        | (closeability_borderline_mask > 0.5)
        | (batch["is_counterfactual"].to(device=device, dtype=torch.float32) > 0.5)
    )
    if bool(getattr(args, "corrective_focus_only", False)):
        corrective_mask = corrective_mask & corrective_focus
    corrective_dx_label = batch["corrective_dx_label"].to(device=device)
    corrective_dy_label = batch["corrective_dy_label"].to(device=device)
    corrective_dz_label = batch["corrective_dz_label"].to(device=device)
    corrective_dyaw_label = batch["corrective_dyaw_label"].to(device=device)
    corrective_dyaw_coarse_label = batch["corrective_dyaw_coarse_label"].to(device=device)
    corrective_dyaw_residual_target = batch["corrective_dyaw_sym_residual"].to(device=device, dtype=torch.float32)
    is_counterfactual = batch["is_counterfactual"].to(device=device, dtype=torch.float32) > 0.5

    metric_l1 = F.smooth_l1_loss(pred_metrics, teacher_metrics, reduction="none")
    metric_l1[:, 0] *= float(args.lambda_xy)
    metric_l1[:, 1] *= float(args.lambda_z)
    metric_l1[:, 2] *= float(args.lambda_yaw)
    metric_loss = metric_l1.mean(dim=-1)
    band_loss = F.cross_entropy(band_logits, teacher_band, reduction="none")
    ready_loss = F.binary_cross_entropy_with_logits(ready_logit, teacher_ready, reduction="none")
    ready_loss = torch.where(is_counterfactual, torch.zeros_like(ready_loss), ready_loss)

    pred_score = 0.45 * pred_metrics[:, 0] + 0.30 * pred_metrics[:, 1] + 0.25 * pred_metrics[:, 2]
    teacher_score = 0.45 * teacher_metrics[:, 0] + 0.30 * teacher_metrics[:, 1] + 0.25 * teacher_metrics[:, 2]
    score_loss = F.smooth_l1_loss(pred_score, teacher_score, reduction="none")
    derived_progress_logit = teacher_score.detach() - pred_score
    if progress_logit is None:
        progress_logit = derived_progress_logit
    progress_loss = F.binary_cross_entropy_with_logits(progress_logit, progress_label, reduction="none")
    pos = progress_mask * (progress_label > 0.5).float()
    neg = progress_mask * (progress_label <= 0.5).float()
    pos_w = 0.5 * torch.clamp(progress_mask.sum(), min=1.0) / torch.clamp(pos.sum(), min=1.0)
    neg_w = 0.5 * torch.clamp(progress_mask.sum(), min=1.0) / torch.clamp(neg.sum(), min=1.0)
    progress_weight = pos * pos_w + neg * neg_w
    progress_loss = progress_loss * progress_weight
    axis_loss = torch.zeros_like(metric_loss)
    if axis_block_logits is not None and torch.any(boundary_mask):
        axis_loss_full = F.cross_entropy(axis_block_logits, axis_label, reduction="none")
        axis_loss = axis_loss_full * boundary_mask.float()
    closeability_loss = torch.zeros_like(metric_loss)
    if closeability_logit is not None:
        closeability_loss = F.binary_cross_entropy_with_logits(closeability_logit, closeability_label, reduction="none")
        closeability_loss = closeability_loss * (1.0 + float(args.closeability_pos_weight) * closeability_label)
        # Borderline rows are especially important: they are the supervision
        # closest to "would be closeable soon" without requiring true ready.
        closeability_loss = closeability_loss * (1.0 + float(args.closeability_borderline_boost) * closeability_borderline_mask)
    corrective_loss = torch.zeros_like(metric_loss)
    if torch.any(corrective_mask):
        parts = []
        if corrective_dx_logits is not None:
            w = getattr(args, "corrective_class_weights_dx", None)
            parts.append(F.cross_entropy(corrective_dx_logits, corrective_dx_label, reduction="none", weight=w.to(device=device) if w is not None else None))
        if corrective_dy_logits is not None:
            w = getattr(args, "corrective_class_weights_dy", None)
            parts.append(F.cross_entropy(corrective_dy_logits, corrective_dy_label, reduction="none", weight=w.to(device=device) if w is not None else None))
        if corrective_dz_logits is not None:
            w = getattr(args, "corrective_class_weights_dz", None)
            parts.append(F.cross_entropy(corrective_dz_logits, corrective_dz_label, reduction="none", weight=w.to(device=device) if w is not None else None))
        dyaw_parts = []
        if corrective_dyaw_logits is not None:
            w = getattr(args, "corrective_class_weights_dyaw", None)
            dyaw_parts.append(
                F.cross_entropy(
                    corrective_dyaw_logits,
                    corrective_dyaw_coarse_label,
                    reduction="none",
                    weight=w.to(device=device) if w is not None else None,
                )
            )
        if corrective_dyaw_residual_pred is not None:
            dyaw_parts.append(F.smooth_l1_loss(corrective_dyaw_residual_pred, corrective_dyaw_residual_target, reduction="none"))
        if dyaw_parts:
            dyaw_loss = torch.stack(dyaw_parts, dim=0).mean(dim=0)
            dyaw_loss = dyaw_loss * 0.35
            parts.append(dyaw_loss)
        if parts:
            corrective_loss = torch.stack(parts, dim=0).mean(dim=0) * corrective_mask.float()
            if bool(getattr(args, "corrective_focus_only", False)):
                corrective_loss = corrective_loss * float(getattr(args, "corrective_focus_boost", 1.0))

    total = (
        metric_loss
        + float(args.lambda_band) * band_loss
        + float(args.lambda_ready) * ready_loss
        + float(args.lambda_progress) * progress_loss
        + float(args.lambda_axis) * axis_loss
        + float(args.lambda_closeability) * closeability_loss
        + float(args.lambda_corrective) * corrective_loss
        + float(args.lambda_score) * score_loss
    )
    loss = (total * sample_weight).sum() / torch.clamp(sample_weight.sum(), min=1e-6)
    if float(args.lambda_far_negative_ready) > 0.0 and torch.any(far_negative):
        neg = F.softplus(ready_logit[far_negative] + float(args.ready_neg_margin_logit))
        neg_w = sample_weight[far_negative]
        loss = loss + float(args.lambda_far_negative_ready) * (neg * neg_w).sum() / torch.clamp(neg_w.sum(), min=1e-6)
    return loss, out


def compute_pairwise_loss(model, batch, device, args):
    cur = forward_model(model, prefixed_batch(batch, "cur_"), device)
    prev = forward_model(model, prefixed_batch(batch, "prev_"), device)
    cur_score = cur.get("closeness_score")
    prev_score = prev.get("closeness_score")
    if cur_score is None or prev_score is None:
        raise RuntimeError("model did not return closeness_score for pairwise ranking")
    label = batch["cur_pair_label"].to(device=device, dtype=torch.float32)
    sample_weight = batch["cur_sample_weight"].to(device=device, dtype=torch.float32)
    # label=1 means current state is closer to ready than the previous state.
    direction = label * 2.0 - 1.0
    margin = direction * (cur_score - prev_score)
    loss_vec = F.softplus(float(args.pair_rank_margin) - margin)
    pos = (label > 0.5).float()
    neg = (label <= 0.5).float()
    total = torch.clamp(pos.sum() + neg.sum(), min=1.0)
    pos_w = 0.5 * total / torch.clamp(pos.sum(), min=1.0)
    neg_w = 0.5 * total / torch.clamp(neg.sum(), min=1.0)
    class_w = pos * pos_w + neg * neg_w
    loss = (loss_vec * sample_weight * class_w).sum() / torch.clamp((sample_weight * class_w).sum(), min=1e-6)
    return loss, {
        "pair_margin": margin.detach(),
        "pair_label": label.detach(),
        "cur_closeness_score": cur_score.detach(),
        "prev_closeness_score": prev_score.detach(),
    }


@torch.no_grad()
def evaluate(model, loader, device, args):
    model.eval()
    sums: dict[str, float] = {}
    counts: dict[str, float] = {}

    def add(key: str, value: torch.Tensor, weight: torch.Tensor | None = None):
        v = value.detach().float()
        if weight is None:
            sums[key] = sums.get(key, 0.0) + float(v.sum().item())
            counts[key] = counts.get(key, 0.0) + float(v.numel())
        else:
            w = weight.detach().float()
            sums[key] = sums.get(key, 0.0) + float((v * w).sum().item())
            counts[key] = counts.get(key, 0.0) + float(w.sum().item())

    loss_sum = 0.0
    rows = 0
    source_counts: dict[str, int] = {}
    closeability_probs_all: list[float] = []
    closeability_labels_all: list[float] = []
    corrective_focus_correct_sum = 0.0
    corrective_focus_count = 0.0
    corrective_focus_axis_sums: dict[str, float] = {}
    corrective_focus_axis_counts: dict[str, float] = {}
    for batch in loader:
        loss, out = compute_loss(model, batch, device, args)
        bsz = int(batch["teacher_truth_handoff_ready"].shape[0])
        loss_sum += float(loss.item()) * bsz
        rows += bsz
        pred = out["pred_metrics_norm"]
        band_logits = out["band_logits"]
        progress_logit = out.get("progress_logit")
        axis_block_logits = out.get("axis_block_logits")
        closeability_logit = out.get("closeability_logit")
        corrective_dx_logits = out.get("corrective_dx_logits")
        corrective_dy_logits = out.get("corrective_dy_logits")
        corrective_dz_logits = out.get("corrective_dz_logits")
        corrective_dyaw_logits = out.get("corrective_dyaw_logits")
        corrective_dyaw_legacy_logits = out.get("corrective_dyaw_legacy_logits")
        corrective_dyaw_residual_pred = out.get("corrective_dyaw_residual")
        ready_prob = torch.sigmoid(out["ready_logit"])
        teacher = batch["teacher_metrics_norm"].to(device=device, dtype=torch.float32)
        band = batch["teacher_band_label"].to(device=device)
        progress_label = batch["progress_label"].to(device=device, dtype=torch.float32)
        progress_mask = batch["progress_mask"].to(device=device, dtype=torch.float32)
        far_negative = batch["far_negative_mask"].to(device=device, dtype=torch.float32) > 0.5
        boundary = batch["boundary_mask"].to(device=device, dtype=torch.float32) > 0.5
        axis_label = batch["axis_block_label"].to(device=device)
        closeability_label = batch["closeability_label"].to(device=device, dtype=torch.float32)
        closeability_borderline_mask = batch["closeability_borderline_mask"].to(device=device, dtype=torch.float32) > 0.5
        corrective_mask = batch["corrective_mask"].to(device=device, dtype=torch.float32) > 0.5
        corrective_focus = (
            (closeability_label > 0.5)
            | (closeability_borderline_mask > 0.5)
            | (batch["is_counterfactual"].to(device=device, dtype=torch.float32) > 0.5)
        )
        corrective_dx_label = batch["corrective_dx_label"].to(device=device)
        corrective_dy_label = batch["corrective_dy_label"].to(device=device)
        corrective_dz_label = batch["corrective_dz_label"].to(device=device)
        corrective_dyaw_label = batch["corrective_dyaw_label"].to(device=device)
        corrective_dyaw_coarse_label = batch["corrective_dyaw_coarse_label"].to(device=device)
        corrective_dyaw_residual_target = batch["corrective_dyaw_sym_residual"].to(device=device, dtype=torch.float32)

        abs_err = torch.abs(pred - teacher)
        add("mae_xy_norm", abs_err[:, 0])
        add("mae_z_norm", abs_err[:, 1])
        add("mae_yaw_norm", abs_err[:, 2])
        add("boundary_mae_xy_norm", abs_err[:, 0], boundary.float())
        add("boundary_mae_z_norm", abs_err[:, 1], boundary.float())
        add("boundary_mae_yaw_norm", abs_err[:, 2], boundary.float())
        pred_band = band_logits.argmax(dim=-1)
        add("band_acc", (pred_band == band).float())
        pred_axis = axis_block_logits.argmax(dim=-1) if axis_block_logits is not None else pred.argmax(dim=-1)
        add("axis_block_acc", (pred_axis == axis_label).float(), boundary.float())
        pred_score = 0.45 * pred[:, 0] + 0.30 * pred[:, 1] + 0.25 * pred[:, 2]
        teacher_score = 0.45 * teacher[:, 0] + 0.30 * teacher[:, 1] + 0.25 * teacher[:, 2]
        derived_progress_logit = teacher_score - pred_score
        if progress_logit is None:
            progress_logit = derived_progress_logit
        progress_pred = (torch.sigmoid(progress_logit) > 0.5).float()
        add("progress_acc", (progress_pred == progress_label).float(), progress_mask)
        pos_mask = progress_mask * (progress_label > 0.5).float()
        neg_mask = progress_mask * (progress_label <= 0.5).float()
        add("progress_pos_recall", (progress_pred > 0.5).float(), pos_mask)
        add("progress_neg_recall", (progress_pred <= 0.5).float(), neg_mask)
        derived_progress_pred = (torch.sigmoid(derived_progress_logit) > 0.5).float()
        add("derived_progress_acc", (derived_progress_pred == progress_label).float(), progress_mask)
        add("derived_progress_pos_recall", (derived_progress_pred > 0.5).float(), pos_mask)
        add("derived_progress_neg_recall", (derived_progress_pred <= 0.5).float(), neg_mask)
        add("progress_logit_abs_mean", torch.abs(progress_logit), progress_mask)
        add("far_negative_ready_prob_mean", ready_prob, far_negative.float())
        add("ready_prob_mean", ready_prob)
        add("pred_release_band_rate", ((pred[:, 0] < 1.0) & (pred[:, 1] < 1.0) & (pred[:, 2] < 1.0)).float())
        if closeability_logit is not None:
            closeability_prob = torch.sigmoid(closeability_logit)
            closeability_pred = (closeability_prob > 0.5).float()
            add("closeability_acc", (closeability_pred == closeability_label).float())
            close_pos = (closeability_label > 0.5).float()
            close_neg = (closeability_label <= 0.5).float()
            add("closeability_pos_recall", (closeability_pred > 0.5).float(), close_pos)
            add("closeability_neg_recall", (closeability_pred <= 0.5).float(), close_neg)
            add("closeability_prob_mean", closeability_prob)
            add("closeability_borderline_prob_mean", closeability_prob, closeability_borderline_mask.float())
            closeability_probs_all.extend(closeability_prob.detach().cpu().float().numpy().tolist())
            closeability_labels_all.extend(closeability_label.detach().cpu().float().numpy().tolist())
        if corrective_dx_logits is not None:
            dx_ok = (corrective_dx_logits.argmax(dim=-1) == corrective_dx_label).float()
            add("corrective_dx_acc", dx_ok, corrective_mask.float())
            add("corrective_dx_focus_acc", dx_ok, (corrective_focus & corrective_mask).float())
        if corrective_dy_logits is not None:
            dy_ok = (corrective_dy_logits.argmax(dim=-1) == corrective_dy_label).float()
            add("corrective_dy_acc", dy_ok, corrective_mask.float())
            add("corrective_dy_focus_acc", dy_ok, (corrective_focus & corrective_mask).float())
        if corrective_dz_logits is not None:
            dz_ok = (corrective_dz_logits.argmax(dim=-1) == corrective_dz_label).float()
            add("corrective_dz_acc", dz_ok, corrective_mask.float())
            add("corrective_dz_focus_acc", dz_ok, (corrective_focus & corrective_mask).float())
        if corrective_dyaw_logits is not None:
            coarse_probs = torch.softmax(corrective_dyaw_logits, dim=-1)
            coarse_pred = coarse_probs.argmax(dim=-1)
            coarse_ok = (coarse_pred == corrective_dyaw_coarse_label).float()
            add("corrective_dyaw_aux_acc", coarse_ok, corrective_mask.float())
            add("corrective_dyaw_aux_focus_acc", coarse_ok, (corrective_focus & corrective_mask).float())
            add("corrective_dyaw_aux_prob_mean", coarse_probs.amax(dim=-1), corrective_mask.float())
            add("corrective_dyaw_aux_prob_focus_mean", coarse_probs.amax(dim=-1), (corrective_focus & corrective_mask).float())
            if corrective_dyaw_residual_pred is not None:
                dyaw_resid_mae = torch.abs(corrective_dyaw_residual_pred - corrective_dyaw_residual_target)
                add("corrective_dyaw_residual_mae", dyaw_resid_mae, corrective_mask.float())
                add("corrective_dyaw_residual_focus_mae", dyaw_resid_mae, (corrective_focus & corrective_mask).float())
            if corrective_dyaw_legacy_logits is not None:
                raw_dyaw_ok = (corrective_dyaw_legacy_logits.argmax(dim=-1) == corrective_dyaw_label).float()
                add("corrective_dyaw_raw_acc", raw_dyaw_ok, corrective_mask.float())
                add("corrective_dyaw_raw_focus_acc", raw_dyaw_ok, (corrective_focus & corrective_mask).float())
        for name in batch["source_name"]:
            source_counts[str(name)] = source_counts.get(str(name), 0) + 1

    out_metrics = {"loss": loss_sum / max(rows, 1), "rows": rows, "source_counts": source_counts}
    for key, value in sums.items():
        out_metrics[key] = value / max(counts.get(key, 0.0), 1e-6)
    out_metrics["progress_balanced_acc"] = 0.5 * (
        out_metrics.get("progress_pos_recall", 0.0) + out_metrics.get("progress_neg_recall", 0.0)
    )
    out_metrics["derived_progress_balanced_acc"] = 0.5 * (
        out_metrics.get("derived_progress_pos_recall", 0.0) + out_metrics.get("derived_progress_neg_recall", 0.0)
    )
    out_metrics["closeability_balanced_acc"] = 0.5 * (
        out_metrics.get("closeability_pos_recall", 0.0) + out_metrics.get("closeability_neg_recall", 0.0)
    )
    close_probs_np = np.asarray(closeability_probs_all, dtype=np.float32)
    close_labels_np = np.asarray(closeability_labels_all, dtype=np.float32) > 0.5
    if close_probs_np.size > 0 and np.any(close_labels_np) and np.any(~close_labels_np):
        thresholds = np.unique(np.quantile(close_probs_np, np.linspace(0.0, 1.0, 101)))
        best_close = {
            "threshold": 0.5,
            "balanced_acc": 0.5 * (
                float(np.mean(close_probs_np[close_labels_np] > 0.5)) + float(np.mean(close_probs_np[~close_labels_np] <= 0.5))
            ),
            "pos_recall": float(np.mean(close_probs_np[close_labels_np] > 0.5)),
            "neg_recall": float(np.mean(close_probs_np[~close_labels_np] <= 0.5)),
        }
        for th in thresholds.tolist():
            pred_np = close_probs_np > float(th)
            p_recall = float(np.mean(pred_np[close_labels_np])) if np.any(close_labels_np) else 0.0
            n_recall = float(np.mean(~pred_np[~close_labels_np])) if np.any(~close_labels_np) else 0.0
            ba = 0.5 * (p_recall + n_recall)
            if ba > float(best_close["balanced_acc"]):
                best_close = {
                    "threshold": float(th),
                    "balanced_acc": float(ba),
                    "pos_recall": float(p_recall),
                    "neg_recall": float(n_recall),
                }
        out_metrics["closeability_calibrated_threshold"] = float(best_close["threshold"])
        out_metrics["closeability_calibrated_balanced_acc"] = float(best_close["balanced_acc"])
        out_metrics["closeability_calibrated_pos_recall"] = float(best_close["pos_recall"])
        out_metrics["closeability_calibrated_neg_recall"] = float(best_close["neg_recall"])
    if "closeability_calibrated_balanced_acc" not in out_metrics:
        out_metrics["closeability_calibrated_threshold"] = 0.5
        out_metrics["closeability_calibrated_balanced_acc"] = out_metrics["closeability_balanced_acc"]
        out_metrics["closeability_calibrated_pos_recall"] = out_metrics.get("closeability_pos_recall", 0.0)
        out_metrics["closeability_calibrated_neg_recall"] = out_metrics.get("closeability_neg_recall", 0.0)
    corrective_terms = [
        out_metrics.get("corrective_dx_acc", 0.0),
        out_metrics.get("corrective_dy_acc", 0.0),
        out_metrics.get("corrective_dz_acc", 0.0),
    ]
    out_metrics["corrective_sign_mean_acc"] = float(sum(corrective_terms) / max(len(corrective_terms), 1))
    corrective_focus_terms = [
        out_metrics.get("corrective_dx_focus_acc", 0.0),
        out_metrics.get("corrective_dy_focus_acc", 0.0),
        out_metrics.get("corrective_dz_focus_acc", 0.0),
    ]
    out_metrics["corrective_focus_mean_acc"] = float(sum(corrective_focus_terms) / max(len(corrective_focus_terms), 1))
    out_metrics["corrective_core_sign_mean_acc"] = float(out_metrics["corrective_sign_mean_acc"])
    out_metrics["corrective_core_focus_mean_acc"] = float(out_metrics["corrective_focus_mean_acc"])
    out_metrics["corrective_dyaw_raw_acc"] = float(out_metrics.get("corrective_dyaw_raw_acc", 0.0))
    out_metrics["corrective_dyaw_raw_focus_acc"] = float(out_metrics.get("corrective_dyaw_raw_focus_acc", 0.0))
    if "corrective_dyaw_aux_acc" in out_metrics:
        out_metrics["corrective_dyaw_aux_acc"] = float(out_metrics.get("corrective_dyaw_aux_acc", 0.0))
        out_metrics["corrective_dyaw_aux_focus_acc"] = float(out_metrics.get("corrective_dyaw_aux_focus_acc", 0.0))
        out_metrics["corrective_dyaw_aux_prob_mean"] = float(out_metrics.get("corrective_dyaw_aux_prob_mean", 0.0))
        out_metrics["corrective_dyaw_aux_prob_focus_mean"] = float(out_metrics.get("corrective_dyaw_aux_prob_focus_mean", 0.0))
    if "corrective_dyaw_residual_mae" in out_metrics:
        out_metrics["corrective_dyaw_residual_mae"] = float(out_metrics.get("corrective_dyaw_residual_mae", 0.0))
        out_metrics["corrective_dyaw_residual_focus_mae"] = float(out_metrics.get("corrective_dyaw_residual_focus_mae", 0.0))
    return out_metrics


@torch.no_grad()
def evaluate_pairwise(model, loader, device, args):
    model.eval()
    if loader is None:
        return {
            "pair_rows": 0,
            "pair_acc": 0.0,
            "pair_pos_recall": 0.0,
            "pair_neg_recall": 0.0,
            "pair_balanced_acc": 0.0,
            "pair_margin_mean": 0.0,
        }
    rows = 0
    correct = 0.0
    pos_total = 0.0
    neg_total = 0.0
    pos_correct = 0.0
    neg_correct = 0.0
    margin_sum = 0.0
    score_gap_sum = 0.0
    all_gaps: list[float] = []
    all_labels: list[float] = []
    for batch in loader:
        cur = forward_model(model, prefixed_batch(batch, "cur_"), device)
        prev = forward_model(model, prefixed_batch(batch, "prev_"), device)
        gap = cur["closeness_score"] - prev["closeness_score"]
        label = batch["cur_pair_label"].to(device=device, dtype=torch.float32)
        pred = (gap > 0.0).float()
        ok = (pred == label).float()
        pos = label > 0.5
        neg = ~pos
        n = int(label.numel())
        rows += n
        correct += float(ok.sum().item())
        pos_total += float(pos.float().sum().item())
        neg_total += float(neg.float().sum().item())
        pos_correct += float(ok[pos].sum().item()) if torch.any(pos) else 0.0
        neg_correct += float(ok[neg].sum().item()) if torch.any(neg) else 0.0
        direction = label * 2.0 - 1.0
        margin_sum += float((direction * gap).sum().item())
        score_gap_sum += float(gap.sum().item())
        all_gaps.extend(gap.detach().cpu().float().numpy().tolist())
        all_labels.extend(label.detach().cpu().float().numpy().tolist())
    pos_recall = pos_correct / max(pos_total, 1e-6)
    neg_recall = neg_correct / max(neg_total, 1e-6)
    calibrated = {
        "threshold": 0.0,
        "balanced_acc": 0.5 * (pos_recall + neg_recall),
        "pos_recall": pos_recall,
        "neg_recall": neg_recall,
    }
    if rows > 0:
        gaps_np = np.asarray(all_gaps, dtype=np.float32)
        labels_np = np.asarray(all_labels, dtype=np.float32)
        pos_np = labels_np > 0.5
        neg_np = ~pos_np
        thresholds = np.unique(np.quantile(gaps_np, np.linspace(0.0, 1.0, 101)))
        best = calibrated
        for threshold in thresholds.tolist():
            pred_np = gaps_np > float(threshold)
            p_recall = float(np.mean(pred_np[pos_np])) if np.any(pos_np) else 0.0
            n_recall = float(np.mean(~pred_np[neg_np])) if np.any(neg_np) else 0.0
            ba = 0.5 * (p_recall + n_recall)
            if ba > float(best["balanced_acc"]):
                best = {
                    "threshold": float(threshold),
                    "balanced_acc": float(ba),
                    "pos_recall": float(p_recall),
                    "neg_recall": float(n_recall),
                }
        calibrated = best
    return {
        "pair_rows": int(rows),
        "pair_acc": correct / max(rows, 1),
        "pair_pos_recall": pos_recall,
        "pair_neg_recall": neg_recall,
        "pair_balanced_acc": 0.5 * (pos_recall + neg_recall),
        "pair_calibrated_threshold": float(calibrated["threshold"]),
        "pair_calibrated_balanced_acc": float(calibrated["balanced_acc"]),
        "pair_calibrated_pos_recall": float(calibrated["pos_recall"]),
        "pair_calibrated_neg_recall": float(calibrated["neg_recall"]),
        "pair_margin_mean": margin_sum / max(rows, 1),
        "pair_score_gap_mean": score_gap_sum / max(rows, 1),
    }


def load_model(path: str | None, device):
    model = StudentHandoffStateHeadV2().to(device)
    if path:
        ckpt = torch.load(path, map_location="cpu")
        state = ckpt.get("model_state_dict", ckpt)
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(json.dumps({"event": "loaded_ckpt", "path": path, "missing": sorted(missing), "unexpected": sorted(unexpected)}))
    return model


def save_ckpt(path: Path, model, args, metrics: dict, history: list[dict]):
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_kind": "student_handoff_state_head_v2",
            "alignment_version": "closeability_v3",
            "dataset_npz": str(args.dataset_npz),
            "val_metrics": metrics,
            "history": history,
        },
        path,
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_npz", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--init_ckpt", required=True)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--val_ratio", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--weighted_sampling", action="store_true")
    ap.add_argument("--sampler_weight_power", type=float, default=1.0)
    ap.add_argument("--lambda_xy", type=float, default=1.6)
    ap.add_argument("--lambda_z", type=float, default=0.9)
    ap.add_argument("--lambda_yaw", type=float, default=1.1)
    ap.add_argument("--lambda_band", type=float, default=0.35)
    ap.add_argument("--lambda_ready", type=float, default=0.05)
    ap.add_argument("--lambda_progress", type=float, default=0.25)
    ap.add_argument("--lambda_axis", type=float, default=0.20)
    ap.add_argument("--lambda_closeability", type=float, default=0.25)
    ap.add_argument("--lambda_corrective", type=float, default=0.20)
    ap.add_argument("--lambda_score", type=float, default=0.20)
    ap.add_argument("--lambda_pair_rank", type=float, default=0.0)
    ap.add_argument("--pair_rank_margin", type=float, default=0.25)
    ap.add_argument("--lambda_far_negative_ready", type=float, default=0.15)
    ap.add_argument("--ready_neg_margin_logit", type=float, default=1.0)
    ap.add_argument("--closeability_borderline_boost", type=float, default=1.0)
    ap.add_argument("--closeability_pos_weight", type=float, default=1.5)
    ap.add_argument("--auto_closeability_pos_weight", action="store_true")
    ap.add_argument("--auto_corrective_class_weights", action="store_true")
    ap.add_argument("--corrective_class_weight_power", type=float, default=0.5)
    ap.add_argument("--progress_gate_min", type=float, default=0.60)
    ap.add_argument("--progress_neg_recall_gate_min", type=float, default=0.45)
    ap.add_argument("--pair_gate_min", type=float, default=0.60)
    ap.add_argument("--pair_pos_recall_gate_min", type=float, default=0.45)
    ap.add_argument("--pair_neg_recall_gate_min", type=float, default=0.45)
    ap.add_argument("--closeability_gate_min", type=float, default=0.58)
    ap.add_argument("--closeability_calibrated_gate_min", type=float, default=0.58)
    ap.add_argument("--use_calibrated_closeability_gate", action="store_true")
    ap.add_argument("--corrective_gate_min", type=float, default=0.50)
    ap.add_argument("--corrective_focus_only", action="store_true")
    ap.add_argument("--corrective_focus_boost", type=float, default=2.0)
    ap.add_argument("--corrective_yaw_symmetry_period", type=float, default=1.5707963267948966)
    ap.add_argument("--corrective_yaw_symmetry_eps", type=float, default=1e-3)
    ap.add_argument("--use_pairwise_gate", action="store_true")
    ap.add_argument("--use_calibrated_pair_gate", action="store_true")
    ap.add_argument("--geometry_regression_max", type=float, default=1.10)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = AlignmentV3Dataset(args.dataset_npz)
    train_idx, val_idx, val_eps = split_by_episode(dataset, args.val_ratio, args.seed)
    val_idx = cap_far_negative_val_rows(dataset, val_idx, cap_per_episode=32, seed=args.seed)
    train_set = Subset(dataset, train_idx)
    val_set = Subset(dataset, val_idx)
    train_pair_idx = pair_indices_for_split(dataset, train_idx)
    val_pair_idx = pair_indices_for_split(dataset, val_idx)
    if args.auto_closeability_pos_weight:
        args.closeability_pos_weight = compute_binary_pos_weight(dataset, train_idx, "alignment_v3_closeability_label")
        print(json.dumps({"event": "auto_closeability_pos_weight", "value": float(args.closeability_pos_weight)}))
    if args.auto_corrective_class_weights:
        corrective_train_idx = train_idx
        if bool(args.corrective_focus_only):
            focus_mask = corrective_focus_mask(dataset, train_idx)
            focus_idx = [idx for idx, keep in zip(train_idx, focus_mask.tolist()) if keep]
            if focus_idx:
                corrective_train_idx = focus_idx
                print(
                    json.dumps(
                        {
                            "event": "corrective_focus_train_rows",
                            "rows": int(len(corrective_train_idx)),
                            "train_rows": int(len(train_idx)),
                        }
                    )
                )
        args.corrective_class_weights_dx = compute_class_weights(
            dataset, corrective_train_idx, "alignment_v3_corrective_dx_label", 3, power=float(args.corrective_class_weight_power)
        )
        args.corrective_class_weights_dy = compute_class_weights(
            dataset, corrective_train_idx, "alignment_v3_corrective_dy_label", 3, power=float(args.corrective_class_weight_power)
        )
        args.corrective_class_weights_dz = compute_class_weights(
            dataset, corrective_train_idx, "alignment_v3_corrective_dz_label", 3, power=float(args.corrective_class_weight_power)
        )
        args.corrective_class_weights_dyaw = compute_class_weights(
            dataset, corrective_train_idx, "alignment_v3_corrective_dyaw_coarse_label", 5, power=float(args.corrective_class_weight_power)
        )
        print(
            json.dumps(
                {
                    "event": "auto_corrective_class_weights",
                    "dx": args.corrective_class_weights_dx.tolist(),
                    "dy": args.corrective_class_weights_dy.tolist(),
                    "dz": args.corrective_class_weights_dz.tolist(),
                    "dyaw_coarse": args.corrective_class_weights_dyaw.tolist(),
                }
            )
        )
    if args.weighted_sampling:
        w = np.asarray(dataset.data["sample_weight"][train_idx], dtype=np.float64)
        w = np.clip(w, 1e-6, None) ** float(args.sampler_weight_power)
        sampler = WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double), num_samples=len(train_idx), replacement=True)
        train_loader = DataLoader(train_set, batch_size=args.batch_size, sampler=sampler, shuffle=False, num_workers=0)
    else:
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=0)
    pair_loader = (
        DataLoader(PairwiseAlignmentV3Dataset(dataset, train_pair_idx), batch_size=args.batch_size, shuffle=True, num_workers=0)
        if train_pair_idx
        else None
    )
    val_pair_loader = (
        DataLoader(PairwiseAlignmentV3Dataset(dataset, val_pair_idx), batch_size=args.batch_size, shuffle=False, num_workers=0)
        if val_pair_idx
        else None
    )

    split_info = {
        "train_rows": int(len(train_idx)),
        "val_rows": int(len(val_idx)),
        "train_pair_rows": int(len(train_pair_idx)),
        "val_pair_rows": int(len(val_pair_idx)),
        "val_episode_indices": val_eps,
        "dataset_npz": str(args.dataset_npz),
        "init_ckpt": str(args.init_ckpt),
    }
    (output_dir / "split_info.json").write_text(json.dumps(split_info, indent=2))

    baseline = load_model(args.init_ckpt, device)
    baseline_metrics = evaluate(baseline, val_loader, device, args)
    baseline_metrics.update(evaluate_pairwise(baseline, val_pair_loader, device, args))
    (output_dir / "baseline_val_metrics.json").write_text(json.dumps(baseline_metrics, indent=2))
    del baseline

    model = load_model(args.init_ckpt, device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=1e-4)
    history: list[dict] = []
    best_geometry = None
    best_progress = None
    best_deploy = None

    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        loss_sum = 0.0
        count = 0
        pair_iter = iter(pair_loader) if pair_loader is not None else None
        pair_loss_sum = 0.0
        pair_count = 0
        for batch in train_loader:
            loss, _ = compute_loss(model, batch, device, args)
            pair_loss = None
            if pair_iter is not None and float(args.lambda_pair_rank) > 0.0:
                try:
                    pair_batch = next(pair_iter)
                except StopIteration:
                    pair_iter = iter(pair_loader)
                    pair_batch = next(pair_iter)
                pair_loss, _ = compute_pairwise_loss(model, pair_batch, device, args)
                loss = loss + float(args.lambda_pair_rank) * pair_loss
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at epoch {epoch}: {float(loss.item())}")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            bsz = int(batch["teacher_truth_handoff_ready"].shape[0])
            loss_sum += float(loss.item()) * bsz
            count += bsz
            if pair_loss is not None:
                pbsz = int(pair_batch["cur_pair_label"].shape[0])
                pair_loss_sum += float(pair_loss.item()) * pbsz
                pair_count += pbsz
        val_metrics = evaluate(model, val_loader, device, args)
        val_metrics.update(evaluate_pairwise(model, val_pair_loader, device, args))
        row = {
            "epoch": epoch,
            "train_loss": loss_sum / max(count, 1),
            "train_pair_loss": pair_loss_sum / max(pair_count, 1),
            **val_metrics,
        }
        history.append(row)
        print(json.dumps(row))

        geometry_score = (
            val_metrics.get("boundary_mae_xy_norm", 999.0)
            + val_metrics.get("boundary_mae_z_norm", 999.0)
            + val_metrics.get("boundary_mae_yaw_norm", 999.0)
        )
        rank_progress_metric = val_metrics.get("pair_balanced_acc", val_metrics.get("progress_balanced_acc", 0.0))
        progress_score = -rank_progress_metric
        closeability_metric_prefix = "closeability_calibrated_" if args.use_calibrated_closeability_gate else "closeability_"
        deploy_score = (
            geometry_score
            - 2.0 * rank_progress_metric
            - 0.75 * val_metrics.get(f"{closeability_metric_prefix}balanced_acc", 0.0)
            - 0.50 * val_metrics.get("corrective_sign_mean_acc", 0.0)
            + 5.0 * val_metrics.get("far_negative_ready_prob_mean", 0.0)
        )
        if best_geometry is None or geometry_score < best_geometry:
            best_geometry = geometry_score
            save_ckpt(output_dir / "student_handoff_state_head_v2_alignment_v3_best_geometry.pt", model, args, val_metrics, history)
        if best_progress is None or progress_score < best_progress:
            best_progress = progress_score
            save_ckpt(output_dir / "student_handoff_state_head_v2_alignment_v3_best_progress.pt", model, args, val_metrics, history)
        if best_deploy is None or deploy_score < best_deploy:
            best_deploy = deploy_score
            save_ckpt(output_dir / "student_handoff_state_head_v2_alignment_v3_best_deploy_candidate.pt", model, args, val_metrics, history)

    final_metrics = evaluate(model, val_loader, device, args)
    final_metrics.update(evaluate_pairwise(model, val_pair_loader, device, args))
    save_ckpt(output_dir / "student_handoff_state_head_v2_alignment_v3_final.pt", model, args, final_metrics, history)
    (output_dir / "alignment_v3_train_history.json").write_text(json.dumps(history, indent=2))

    best_deploy_ckpt = torch.load(output_dir / "student_handoff_state_head_v2_alignment_v3_best_deploy_candidate.pt", map_location="cpu")
    candidate_metrics = best_deploy_ckpt.get("val_metrics", {})
    baseline_geom = (
        baseline_metrics.get("boundary_mae_xy_norm", 0.0)
        + baseline_metrics.get("boundary_mae_z_norm", 0.0)
        + baseline_metrics.get("boundary_mae_yaw_norm", 0.0)
    )
    candidate_geom = (
        candidate_metrics.get("boundary_mae_xy_norm", 0.0)
        + candidate_metrics.get("boundary_mae_z_norm", 0.0)
        + candidate_metrics.get("boundary_mae_yaw_norm", 0.0)
    )
    geom_gain = baseline_geom - candidate_geom
    geometry_ok = candidate_geom <= baseline_geom * float(args.geometry_regression_max)
    if args.use_pairwise_gate:
        pair_metric_prefix = "pair_calibrated_" if args.use_calibrated_pair_gate else "pair_"
        progress_ok = (
            candidate_metrics.get(f"{pair_metric_prefix}balanced_acc", 0.0) >= float(args.pair_gate_min)
            and candidate_metrics.get(f"{pair_metric_prefix}pos_recall", 0.0) >= float(args.pair_pos_recall_gate_min)
            and candidate_metrics.get(f"{pair_metric_prefix}neg_recall", 0.0) >= float(args.pair_neg_recall_gate_min)
        )
    else:
        progress_ok = (
            candidate_metrics.get("progress_balanced_acc", 0.0) >= float(args.progress_gate_min)
            and candidate_metrics.get("progress_neg_recall", 0.0) >= float(args.progress_neg_recall_gate_min)
        )
    closeability_metric_prefix = "closeability_calibrated_" if args.use_calibrated_closeability_gate else "closeability_"
    corrective_metric_key = "corrective_focus_mean_acc" if bool(args.corrective_focus_only) else "corrective_sign_mean_acc"
    closeability_gate_min = float(
        args.closeability_calibrated_gate_min if args.use_calibrated_closeability_gate else args.closeability_gate_min
    )
    gate = {
        "decision": "shadow_candidate" if (
            geometry_ok
            and progress_ok
            and candidate_metrics.get(f"{closeability_metric_prefix}balanced_acc", 0.0) >= closeability_gate_min
            and candidate_metrics.get(corrective_metric_key, 0.0) >= float(args.corrective_gate_min)
            and candidate_metrics.get("far_negative_ready_prob_mean", 1.0) < 0.01
            and candidate_metrics.get("pred_release_band_rate", 1.0) <= 0.0
        ) else "offline_blocked",
        "geometry_mae_sum_gain_vs_baseline": float(geom_gain),
        "boundary_geometry_mae_sum_baseline": float(baseline_geom),
        "boundary_geometry_mae_sum_candidate": float(candidate_geom),
        "geometry_ok": bool(geometry_ok),
        "progress_balanced_acc": float(candidate_metrics.get("progress_balanced_acc", 0.0)),
        "progress_neg_recall": float(candidate_metrics.get("progress_neg_recall", 0.0)),
        "progress_gate_min": float(args.progress_gate_min),
        "progress_neg_recall_gate_min": float(args.progress_neg_recall_gate_min),
        "use_pairwise_gate": bool(args.use_pairwise_gate),
        "use_calibrated_pair_gate": bool(args.use_calibrated_pair_gate),
        "pair_balanced_acc": float(candidate_metrics.get("pair_balanced_acc", 0.0)),
        "pair_pos_recall": float(candidate_metrics.get("pair_pos_recall", 0.0)),
        "pair_neg_recall": float(candidate_metrics.get("pair_neg_recall", 0.0)),
        "pair_calibrated_threshold": float(candidate_metrics.get("pair_calibrated_threshold", 0.0)),
        "pair_calibrated_balanced_acc": float(candidate_metrics.get("pair_calibrated_balanced_acc", 0.0)),
        "pair_calibrated_pos_recall": float(candidate_metrics.get("pair_calibrated_pos_recall", 0.0)),
        "pair_calibrated_neg_recall": float(candidate_metrics.get("pair_calibrated_neg_recall", 0.0)),
        "pair_gate_min": float(args.pair_gate_min),
        "pair_pos_recall_gate_min": float(args.pair_pos_recall_gate_min),
        "pair_neg_recall_gate_min": float(args.pair_neg_recall_gate_min),
        "progress_ok": bool(progress_ok),
        "closeability_balanced_acc": float(candidate_metrics.get("closeability_balanced_acc", 0.0)),
        "closeability_calibrated_threshold": float(candidate_metrics.get("closeability_calibrated_threshold", 0.5)),
        "closeability_calibrated_balanced_acc": float(candidate_metrics.get("closeability_calibrated_balanced_acc", 0.0)),
        "closeability_calibrated_pos_recall": float(candidate_metrics.get("closeability_calibrated_pos_recall", 0.0)),
        "closeability_calibrated_neg_recall": float(candidate_metrics.get("closeability_calibrated_neg_recall", 0.0)),
        "closeability_gate_min": float(args.closeability_gate_min),
        "closeability_calibrated_gate_min": float(args.closeability_calibrated_gate_min),
        "use_calibrated_closeability_gate": bool(args.use_calibrated_closeability_gate),
        "corrective_sign_mean_acc": float(candidate_metrics.get("corrective_sign_mean_acc", 0.0)),
        "corrective_focus_mean_acc": float(candidate_metrics.get("corrective_focus_mean_acc", 0.0)),
        "corrective_metric_key": corrective_metric_key,
        "corrective_gate_min": float(args.corrective_gate_min),
        "far_negative_ready_prob_mean": float(candidate_metrics.get("far_negative_ready_prob_mean", 0.0)),
        "baseline_val_metrics": baseline_metrics,
        "candidate_val_metrics": candidate_metrics,
    }
    (output_dir / "alignment_v3_gate_report.json").write_text(json.dumps(gate, indent=2))
    print(json.dumps(gate, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
