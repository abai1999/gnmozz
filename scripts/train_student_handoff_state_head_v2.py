"""
Train the phase-1 student handoff-state head v2.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler

from prismatic.models.student_handoff_state_head_v2 import StudentHandoffStateHeadV2


class HandoffStateDatasetV2(Dataset):
    def __init__(self, npz_path: str):
        arr = np.load(npz_path, allow_pickle=False)
        self.data = {k: np.asarray(arr[k]) for k in arr.files}
        if "episode_index" not in self.data:
            raise RuntimeError("dataset must contain `episode_index` for episode-level split.")
        self.length = int(self.data["teacher_truth_handoff_ready"].shape[0])

    def __len__(self):
        return self.length

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
            "gripper_context": torch.from_numpy(self.data["gripper_context"][idx].astype(np.float32)),
            "proxy_current_delta_basin_target": torch.from_numpy(self.data["proxy_current_delta_basin_target"][idx].astype(np.float32)),
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
            "near_xy_hard": torch.tensor(float(self.data["near_xy_hard"][idx]), dtype=torch.float32),
            "broad_xy_recovery": torch.tensor(float(self.data.get("broad_xy_recovery", self.data["near_xy_hard"])[idx]), dtype=torch.float32),
            "near_yaw_hard": torch.tensor(float(self.data["near_yaw_hard"][idx]), dtype=torch.float32),
            "near_coupled": torch.tensor(float(self.data["near_coupled"][idx]), dtype=torch.float32),
            "ready_support": torch.tensor(float(self.data["ready_support"][idx]), dtype=torch.float32),
            "yaw_needed_v1": torch.tensor(
                float(self.data.get("yaw_needed_v1", self.data["near_yaw_hard"])[idx]),
                dtype=torch.float32,
            ),
            "xy_block_v1": torch.tensor(
                float(self.data.get("xy_block_v1", self.data["near_xy_hard"])[idx]),
                dtype=torch.float32,
            ),
            "runtime_handoff_metric_valid": torch.tensor(float(self.data["runtime_handoff_metric_valid"][idx]), dtype=torch.float32),
            "runtime_handoff_ready": torch.tensor(float(self.data["runtime_handoff_ready"][idx]), dtype=torch.float32),
            "focus_window_mask_v2": torch.tensor(float(self.data.get("focus_window_mask_v2", np.zeros((self.length,), dtype=np.float32))[idx]), dtype=torch.float32),
            "teacher_ready_exact_mask_v2": torch.tensor(float(self.data.get("teacher_ready_exact_mask_v2", np.zeros((self.length,), dtype=np.float32))[idx]), dtype=torch.float32),
            "boundary_band_mask_v2": torch.tensor(float(self.data.get("boundary_band_mask_v2", np.zeros((self.length,), dtype=np.float32))[idx]), dtype=torch.float32),
            "far_negative_mask_v2": torch.tensor(float(self.data.get("far_negative_mask_v2", np.zeros((self.length,), dtype=np.float32))[idx]), dtype=torch.float32),
            "current_profile_hard_negative_v1": torch.tensor(
                float(self.data.get("current_profile_hard_negative_v1", np.zeros((self.length,), dtype=np.float32))[idx]),
                dtype=torch.float32,
            ),
            "source_name": str(self.data.get("source_name", np.full((self.length,), "unknown", dtype="U32"))[idx]),
        }


def split_by_episode(dataset: HandoffStateDatasetV2, val_ratio: float, seed: int):
    ep = np.asarray(dataset.data["episode_index"], dtype=np.int64)
    uniq = np.unique(ep)
    if uniq.size < 2:
        raise RuntimeError(f"need at least 2 unique episodes for split, got {uniq.size}")
    rng = np.random.default_rng(seed)
    teacher_ready = np.asarray(dataset.data["teacher_truth_handoff_ready"], dtype=np.float32) > 0.5
    teacher_band = np.asarray(dataset.data["teacher_band_label"], dtype=np.int64)

    positive_eps = []
    negative_eps = []
    for e in uniq.tolist():
        mask = ep == e
        has_release = bool(np.any(teacher_ready[mask]))
        has_non_support = bool(np.any(teacher_band[mask] > 0))
        if has_release or has_non_support:
            positive_eps.append(e)
        else:
            negative_eps.append(e)

    rng.shuffle(positive_eps)
    rng.shuffle(negative_eps)
    val_n = max(1, int(round(uniq.size * val_ratio)))
    if val_n >= uniq.size:
        val_n = max(1, uniq.size - 1)

    val_positive_n = 0
    if len(positive_eps) >= 2:
        val_positive_n = max(1, int(round(len(positive_eps) * val_ratio)))
        val_positive_n = min(val_positive_n, len(positive_eps) - 1)

    val_negative_n = val_n - val_positive_n
    if val_negative_n >= len(negative_eps) and len(negative_eps) > 0:
        val_negative_n = max(0, len(negative_eps) - 1)
    if val_positive_n + val_negative_n <= 0:
        if len(negative_eps) > 1:
            val_negative_n = 1
        elif len(positive_eps) > 1:
            val_positive_n = 1

    val_set = set(positive_eps[:val_positive_n] + negative_eps[:val_negative_n])
    train_idx = [i for i, e in enumerate(ep.tolist()) if e not in val_set]
    val_idx = [i for i, e in enumerate(ep.tolist()) if e in val_set]
    return train_idx, val_idx


def split_by_fixed_val_episodes(dataset: HandoffStateDatasetV2, val_episodes: list[int]):
    ep = np.asarray(dataset.data["episode_index"], dtype=np.int64)
    uniq = np.unique(ep)
    if uniq.size < 2:
        raise RuntimeError(f"need at least 2 unique episodes for split, got {uniq.size}")
    val_set = {int(x) for x in val_episodes}
    present = sorted(int(x) for x in np.unique(ep[np.isin(ep, list(val_set))]).tolist())
    if not present:
        raise RuntimeError("none of the requested val_episodes are present in the dataset")
    if len(present) >= int(uniq.size):
        raise RuntimeError("val_episodes cover all dataset episodes; no train episodes remain")
    train_idx = [i for i, e in enumerate(ep.tolist()) if e not in val_set]
    val_idx = [i for i, e in enumerate(ep.tolist()) if e in val_set]
    return train_idx, val_idx, present


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
    )


def _forward_model_masked(model, batch, device, mask):
    # Batch tensors from the dataloader live on CPU here; keep the mask on CPU
    # while slicing, then move the sliced tensors to the target device.
    cpu_mask = mask.detach().to(device="cpu")
    return model(
        front_rgb=batch["front_rgb"][cpu_mask].to(device=device, dtype=torch.float32),
        wrist_rgb=batch["wrist_rgb"][cpu_mask].to(device=device, dtype=torch.float32),
        wrist_depth=batch["wrist_depth"][cpu_mask].to(device=device, dtype=torch.float32),
        proprio=batch["proprio"][cpu_mask].to(device=device, dtype=torch.float32),
        gripper_context=batch["gripper_context"][cpu_mask].to(device=device, dtype=torch.float32),
        proxy_current_delta_basin_target=batch["proxy_current_delta_basin_target"][cpu_mask].to(device=device, dtype=torch.float32),
        current_dx_sign=batch["current_dx_sign"][cpu_mask].to(device=device),
        current_dy_sign=batch["current_dy_sign"][cpu_mask].to(device=device),
        current_dyaw_sign=batch["current_dyaw_sign"][cpu_mask].to(device=device),
        basin_distance_bin=batch["basin_distance_bin"][cpu_mask].to(device=device),
        substage_id=batch["substage_id"][cpu_mask].to(device=device),
        contact_state=batch["contact_state"][cpu_mask].to(device=device),
        stage_target_mode=batch["stage_target_mode"][cpu_mask].to(device=device),
    )


def _subset_diag(pred_metrics, teacher_metrics, ready_prob, uncertainty, teacher_ready, teacher_band, subset_mask):
    subset_mask = subset_mask.bool()
    if not torch.any(subset_mask):
        return {
            "rows": 0.0,
            "mae_xy_norm": 0.0,
            "mae_z_norm": 0.0,
            "mae_yaw_norm": 0.0,
            "false_ready_rate": 0.0,
            "xy_in_band_rate": 0.0,
            "z_in_band_rate": 0.0,
            "yaw_in_band_rate": 0.0,
            "release_band_rate": 0.0,
            "ready_prob_mean": 0.0,
            "ready_prob_pos_rate": 0.0,
            "uncertainty_mean": 0.0,
            "teacher_ready_rate": 0.0,
            "teacher_ready_rows": 0.0,
            "teacher_ready_xy_in_band_rate": 0.0,
            "teacher_ready_z_in_band_rate": 0.0,
            "teacher_ready_yaw_in_band_rate": 0.0,
            "teacher_ready_ready_prob_mean": 0.0,
            "band_gt1_rate": 0.0,
        }
    pm = pred_metrics[subset_mask]
    rp = ready_prob[subset_mask]
    uc = uncertainty[subset_mask]
    tr = teacher_ready[subset_mask]
    tb = teacher_band[subset_mask]
    release_band = (pm[:, 0] < 1.0) & (pm[:, 1] < 1.0) & (pm[:, 2] < 1.0)
    teacher_ready_mask = tr > 0.5
    false_ready = (rp > 0.5) & (tr <= 0.5)
    if torch.any(teacher_ready_mask):
        tr_pm = pm[teacher_ready_mask]
        tr_rp = rp[teacher_ready_mask]
        teacher_ready_xy_in_band_rate = float(torch.mean((tr_pm[:, 0] < 1.0).float()).item())
        teacher_ready_z_in_band_rate = float(torch.mean((tr_pm[:, 1] < 1.0).float()).item())
        teacher_ready_yaw_in_band_rate = float(torch.mean((tr_pm[:, 2] < 1.0).float()).item())
        teacher_ready_ready_prob_mean = float(torch.mean(tr_rp).item())
        teacher_ready_rows = float(torch.sum(teacher_ready_mask.float()).item())
    else:
        teacher_ready_xy_in_band_rate = 0.0
        teacher_ready_z_in_band_rate = 0.0
        teacher_ready_yaw_in_band_rate = 0.0
        teacher_ready_ready_prob_mean = 0.0
        teacher_ready_rows = 0.0
    return {
        "rows": float(pm.shape[0]),
        "mae_xy_norm": float(torch.mean(torch.abs(pm[:, 0] - teacher_metrics[subset_mask][:, 0])).item()),
        "mae_z_norm": float(torch.mean(torch.abs(pm[:, 1] - teacher_metrics[subset_mask][:, 1])).item()),
        "mae_yaw_norm": float(torch.mean(torch.abs(pm[:, 2] - teacher_metrics[subset_mask][:, 2])).item()),
        "false_ready_rate": float(torch.mean(false_ready.float()).item()),
        "xy_in_band_rate": float(torch.mean((pm[:, 0] < 1.0).float()).item()),
        "z_in_band_rate": float(torch.mean((pm[:, 1] < 1.0).float()).item()),
        "yaw_in_band_rate": float(torch.mean((pm[:, 2] < 1.0).float()).item()),
        "release_band_rate": float(torch.mean(release_band.float()).item()),
        "ready_prob_mean": float(torch.mean(rp).item()),
        "ready_prob_pos_rate": float(torch.mean((rp > 0.5).float()).item()),
        "uncertainty_mean": float(torch.mean(uc).item()),
        "teacher_ready_rate": float(torch.mean(tr).item()),
        "teacher_ready_rows": teacher_ready_rows,
        "teacher_ready_xy_in_band_rate": teacher_ready_xy_in_band_rate,
        "teacher_ready_z_in_band_rate": teacher_ready_z_in_band_rate,
        "teacher_ready_yaw_in_band_rate": teacher_ready_yaw_in_band_rate,
        "teacher_ready_ready_prob_mean": teacher_ready_ready_prob_mean,
        "band_gt1_rate": float(torch.mean((tb > 0).float()).item()),
    }


def compute_loss(
    model,
    batch,
    device,
    lambda_band: float,
    lambda_ready: float,
    lambda_uncertainty: float,
    lambda_xy: float,
    lambda_z: float,
    lambda_yaw: float,
    lambda_teacher_ready_push: float = 0.0,
    lambda_far_negative_calib: float = 0.0,
    lambda_close_ready_crossing: float = 0.0,
    lambda_z_aware_ready_crossing: float = 0.0,
    lambda_late_ready_logit_lift: float = 0.0,
    lambda_ready_neighborhood_consistency: float = 0.0,
    lambda_far_negative_hard: float = 0.0,
    far_negative_min: float = 2.5,
    crossing_xy_max: float = 1.05,
    crossing_z_max: float = 1.0,
    crossing_yaw_max: float = 1.0,
    z_ready_xy_near_max: float = 1.10,
    z_ready_z_max: float = 0.90,
    z_ready_yaw_max: float = 1.00,
    late_ready_xy_max: float = 1.00,
    late_ready_z_max: float = 1.00,
    late_ready_yaw_max: float = 1.00,
    late_ready_uncertainty_max: float = 0.50,
    late_ready_prob_min: float = 0.40,
    late_ready_prob_max: float = 0.50,
    ready_pos_margin_logit: float = 1.0,
    ready_neg_margin_logit: float = 1.0,
    ready_neighborhood_target_logit: float = 0.0,
    consistency_model=None,
    lambda_consistency_band: float = 0.0,
    lambda_consistency_ready: float = 0.0,
    consistency_sources: tuple[str, ...] = (),
    lambda_current_profile_hard_negative_veto: float = 0.0,
    hard_negative_yaw_margin_norm: float = 1.15,
):
    out = forward_model(model, batch, device)
    pred_metrics = out["pred_metrics_norm"]
    band_logits = out["band_logits"]
    ready_logit = out["ready_logit"]
    uncertainty = out["uncertainty"]

    teacher_metrics = batch["teacher_metrics_norm"].to(device=device, dtype=torch.float32)
    teacher_band = batch["teacher_band_label"].to(device=device)
    teacher_ready = batch["teacher_truth_handoff_ready"].to(device=device, dtype=torch.float32)
    sample_weight = batch["sample_weight"].to(device=device, dtype=torch.float32)
    near_xy_hard = batch["near_xy_hard"].to(device=device, dtype=torch.float32)
    near_yaw_hard = batch["near_yaw_hard"].to(device=device, dtype=torch.float32)
    focus_window_mask_v2 = batch.get("focus_window_mask_v2")
    if focus_window_mask_v2 is None:
        focus_window_mask_v2 = torch.zeros_like(teacher_ready)
    else:
        focus_window_mask_v2 = focus_window_mask_v2.to(device=device, dtype=torch.float32)
    teacher_ready_exact_mask_v2 = batch.get("teacher_ready_exact_mask_v2")
    if teacher_ready_exact_mask_v2 is None:
        teacher_ready_exact_mask_v2 = torch.zeros_like(teacher_ready)
    else:
        teacher_ready_exact_mask_v2 = teacher_ready_exact_mask_v2.to(device=device, dtype=torch.float32)
    boundary_band_mask_v2 = batch.get("boundary_band_mask_v2")
    if boundary_band_mask_v2 is None:
        boundary_band_mask_v2 = torch.zeros_like(teacher_ready)
    else:
        boundary_band_mask_v2 = boundary_band_mask_v2.to(device=device, dtype=torch.float32)
    current_profile_hard_negative_v1 = batch.get("current_profile_hard_negative_v1")
    if current_profile_hard_negative_v1 is None:
        current_profile_hard_negative_v1 = torch.zeros_like(teacher_ready)
    else:
        current_profile_hard_negative_v1 = current_profile_hard_negative_v1.to(device=device, dtype=torch.float32)

    metric_l1 = F.smooth_l1_loss(pred_metrics, teacher_metrics, reduction="none")
    metric_l1[:, 0] *= lambda_xy + near_xy_hard
    metric_l1[:, 1] *= lambda_z
    metric_l1[:, 2] *= lambda_yaw + 0.5 * near_yaw_hard
    metric_loss = metric_l1.mean(dim=-1)
    band_loss = F.cross_entropy(band_logits, teacher_band, reduction="none")
    ready_loss = F.binary_cross_entropy_with_logits(ready_logit, teacher_ready, reduction="none")
    ready_prob = torch.sigmoid(ready_logit)
    # Keep uncertainty as an auxiliary diagnostic unless explicitly weighted.
    confidence_penalty = torch.abs(ready_prob - teacher_ready) * uncertainty
    total = metric_loss + lambda_band * band_loss + lambda_ready * ready_loss + lambda_uncertainty * confidence_penalty
    loss = (total * sample_weight).sum() / torch.clamp(sample_weight.sum(), min=1e-6)

    teacher_ready_mask = teacher_ready > 0.5
    close_ready_crossing_mask = (
        teacher_ready_mask
        & (teacher_metrics[:, 0] <= float(crossing_xy_max))
        & (teacher_metrics[:, 1] <= float(crossing_z_max))
        & (teacher_metrics[:, 2] <= float(crossing_yaw_max))
    )
    z_aware_ready_crossing_mask = (
        teacher_ready_mask
        & (teacher_metrics[:, 0] <= float(z_ready_xy_near_max))
        & (teacher_metrics[:, 1] <= float(z_ready_z_max))
        & (teacher_metrics[:, 2] <= float(z_ready_yaw_max))
    )
    late_ready_logit_lift_mask = (
        teacher_ready_mask
        & (teacher_metrics[:, 0] <= float(late_ready_xy_max))
        & (teacher_metrics[:, 1] <= float(late_ready_z_max))
        & (teacher_metrics[:, 2] <= float(late_ready_yaw_max))
        & (uncertainty <= float(late_ready_uncertainty_max))
        & (ready_prob >= float(late_ready_prob_min))
        & (ready_prob < float(late_ready_prob_max))
    )
    far_negative_mask = (
        (teacher_ready <= 0.5)
        & (teacher_band == 0)
        & torch.any(teacher_metrics >= float(far_negative_min), dim=-1)
    )
    current_profile_hard_negative_mask = (current_profile_hard_negative_v1 > 0.5) & (teacher_ready <= 0.5)
    ready_neighborhood_mask = (
        ((focus_window_mask_v2 > 0.5) | (teacher_ready_exact_mask_v2 > 0.5) | (boundary_band_mask_v2 > 0.5))
        & (teacher_metrics[:, 0] <= float(z_ready_xy_near_max))
        & (teacher_metrics[:, 1] <= float(crossing_z_max))
        & (teacher_metrics[:, 2] <= float(crossing_yaw_max))
    )
    if lambda_teacher_ready_push > 0.0 and torch.any(teacher_ready_mask):
        # Explicitly pull ready_logit upward on true teacher-ready windows.
        push_loss = F.softplus(-ready_logit[teacher_ready_mask])
        push_weight = sample_weight[teacher_ready_mask]
        loss = loss + lambda_teacher_ready_push * (push_loss * push_weight).sum() / torch.clamp(push_weight.sum(), min=1e-6)
    if lambda_far_negative_calib > 0.0 and torch.any(far_negative_mask):
        # Explicitly suppress false positive ready on clearly far / not-ready windows.
        neg_loss = F.softplus(ready_logit[far_negative_mask])
        neg_weight = sample_weight[far_negative_mask]
        loss = loss + lambda_far_negative_calib * (neg_loss * neg_weight).sum() / torch.clamp(neg_weight.sum(), min=1e-6)
    if lambda_close_ready_crossing > 0.0 and torch.any(close_ready_crossing_mask):
        # Push teacher-ready boundary rows decisively above the close-ready threshold.
        crossing_loss = F.softplus(float(ready_pos_margin_logit) - ready_logit[close_ready_crossing_mask])
        crossing_weight = sample_weight[close_ready_crossing_mask]
        loss = loss + lambda_close_ready_crossing * (crossing_loss * crossing_weight).sum() / torch.clamp(crossing_weight.sum(), min=1e-6)
    if lambda_z_aware_ready_crossing > 0.0 and torch.any(z_aware_ready_crossing_mask):
        # When teacher-ready windows already have z/yaw in-band and xy is only near the edge,
        # push ready_logit upward more aggressively so close-ready can cross earlier.
        z_crossing_loss = F.softplus(float(ready_pos_margin_logit) - ready_logit[z_aware_ready_crossing_mask])
        z_crossing_weight = sample_weight[z_aware_ready_crossing_mask]
        loss = loss + lambda_z_aware_ready_crossing * (z_crossing_loss * z_crossing_weight).sum() / torch.clamp(z_crossing_weight.sum(), min=1e-6)
    if lambda_late_ready_logit_lift > 0.0 and torch.any(late_ready_logit_lift_mask):
        # Specifically lift teacher-ready boundary rows that already satisfy geometry
        # but still stall with ready_prob in the 0.4x band.
        late_lift_loss = F.softplus(float(ready_pos_margin_logit) - ready_logit[late_ready_logit_lift_mask])
        late_lift_weight = sample_weight[late_ready_logit_lift_mask]
        loss = loss + lambda_late_ready_logit_lift * (late_lift_loss * late_lift_weight).sum() / torch.clamp(late_lift_weight.sum(), min=1e-6)
    if lambda_far_negative_hard > 0.0 and torch.any(far_negative_mask):
        # Use a stronger negative margin so far-negative rows stay safely below release-ready.
        hard_neg_loss = F.softplus(ready_logit[far_negative_mask] + float(ready_neg_margin_logit))
        hard_neg_weight = sample_weight[far_negative_mask]
        loss = loss + lambda_far_negative_hard * (hard_neg_loss * hard_neg_weight).sum() / torch.clamp(hard_neg_weight.sum(), min=1e-6)
    if lambda_current_profile_hard_negative_veto > 0.0 and torch.any(current_profile_hard_negative_mask):
        # Current-profile hard negatives are near the close window but unsafe to release.
        # Keep ready_logit below the threshold and require yaw to remain out-of-band
        # so yaw-block cases (e.g. ep014) are not learned as release-ready.
        hn = current_profile_hard_negative_mask
        ready_veto_loss = F.softplus(ready_logit[hn] + float(ready_neg_margin_logit))
        yaw_veto_loss = F.softplus(float(hard_negative_yaw_margin_norm) - pred_metrics[hn, 2])
        band_veto_loss = F.cross_entropy(
            band_logits[hn],
            torch.zeros_like(teacher_band[hn]),
            reduction="none",
        )
        veto_loss = ready_veto_loss + yaw_veto_loss + 0.25 * band_veto_loss
        veto_weight = sample_weight[hn]
        loss = loss + lambda_current_profile_hard_negative_veto * (veto_loss * veto_weight).sum() / torch.clamp(veto_weight.sum(), min=1e-6)
    if lambda_ready_neighborhood_consistency > 0.0 and torch.any(ready_neighborhood_mask):
        # Smooth the ready neighborhood toward a modest threshold-crossing logit instead of
        # globally pushing all ready probabilities upward.
        neighborhood_loss = F.smooth_l1_loss(
            ready_logit[ready_neighborhood_mask],
            torch.full_like(ready_logit[ready_neighborhood_mask], float(ready_neighborhood_target_logit)),
            reduction="none",
        )
        neighborhood_weight = sample_weight[ready_neighborhood_mask]
        loss = loss + lambda_ready_neighborhood_consistency * (neighborhood_loss * neighborhood_weight).sum() / torch.clamp(neighborhood_weight.sum(), min=1e-6)

    if (
        consistency_model is not None
        and (lambda_consistency_band > 0.0 or lambda_consistency_ready > 0.0)
        and consistency_sources
    ):
        source_names = batch["source_name"]
        anchor_mask_list = [name in consistency_sources for name in source_names]
        if any(anchor_mask_list):
            anchor_mask = torch.tensor(anchor_mask_list, device=device, dtype=torch.bool)
            with torch.no_grad():
                teacher_out = _forward_model_masked(consistency_model, batch, device, anchor_mask)
            anchor_weight = sample_weight[anchor_mask]
            cons_terms = torch.zeros_like(anchor_weight)
            if lambda_consistency_band > 0.0:
                teacher_prob = torch.softmax(teacher_out["band_logits"], dim=-1)
                cons_band = F.kl_div(
                    F.log_softmax(band_logits[anchor_mask], dim=-1),
                    teacher_prob,
                    reduction="none",
                ).sum(dim=-1)
                cons_terms = cons_terms + lambda_consistency_band * cons_band
            if lambda_consistency_ready > 0.0:
                teacher_ready_prob = torch.sigmoid(teacher_out["ready_logit"])
                cons_ready = F.binary_cross_entropy_with_logits(
                    ready_logit[anchor_mask],
                    teacher_ready_prob,
                    reduction="none",
                )
                cons_terms = cons_terms + lambda_consistency_ready * cons_ready
            if torch.any(anchor_weight > 0):
                loss = loss + (cons_terms * anchor_weight).sum() / torch.clamp(anchor_weight.sum(), min=1e-6)

    with torch.no_grad():
        pred_band = band_logits.argmax(dim=-1)
        band_acc = torch.mean((pred_band == teacher_band).float())
        ready_pred = (ready_prob > 0.5).float()
        tp = torch.sum((ready_pred > 0.5) & (teacher_ready > 0.5)).float()
        fp = torch.sum((ready_pred > 0.5) & (teacher_ready <= 0.5)).float()
        fn = torch.sum((ready_pred <= 0.5) & (teacher_ready > 0.5)).float()
        precision = tp / torch.clamp(tp + fp, min=1.0)
        recall = tp / torch.clamp(tp + fn, min=1.0)
        f1 = (2.0 * precision * recall) / torch.clamp(precision + recall, min=1e-6)
        false_ready = torch.mean(((ready_pred > 0.5) & (teacher_ready <= 0.5)).float())
        mae = torch.mean(torch.abs(pred_metrics - teacher_metrics), dim=0)
        uncertain_false_ready = torch.mean(
            torch.where((ready_pred > 0.5) & (teacher_ready <= 0.5), uncertainty, torch.zeros_like(uncertainty))
        )
        if torch.any(teacher_ready_mask):
            teacher_ready_xy_in_band = torch.mean((pred_metrics[teacher_ready_mask, 0] < 1.0).float())
            teacher_ready_z_in_band = torch.mean((pred_metrics[teacher_ready_mask, 1] < 1.0).float())
            teacher_ready_yaw_in_band = torch.mean((pred_metrics[teacher_ready_mask, 2] < 1.0).float())
            teacher_ready_prob_pos = torch.mean((ready_prob[teacher_ready_mask] > 0.5).float())
            teacher_ready_prob_mean = torch.mean(ready_prob[teacher_ready_mask])
            teacher_ready_uncertainty_mean = torch.mean(uncertainty[teacher_ready_mask])
            teacher_ready_rows = float(torch.sum(teacher_ready_mask.float()).item())
        else:
            zero = torch.zeros((), device=teacher_ready.device)
            teacher_ready_xy_in_band = zero
            teacher_ready_z_in_band = zero
            teacher_ready_yaw_in_band = zero
            teacher_ready_prob_pos = zero
            teacher_ready_prob_mean = zero
            teacher_ready_uncertainty_mean = zero
            teacher_ready_rows = 0.0
        pred_release_band = (
            (pred_metrics[:, 0] < 1.0)
            & (pred_metrics[:, 1] < 1.0)
            & (pred_metrics[:, 2] < 1.0)
        ).float()
        very_near_diag = _subset_diag(
            pred_metrics,
            teacher_metrics,
            ready_prob,
            uncertainty,
            teacher_ready,
            teacher_band,
            teacher_band > 0,
        )
        ready_support_diag = _subset_diag(
            pred_metrics,
            teacher_metrics,
            ready_prob,
            uncertainty,
            teacher_ready,
            teacher_band,
            teacher_ready_mask,
        )
    return loss, {
        "mae_xy_norm": float(mae[0].item()),
        "mae_z_norm": float(mae[1].item()),
        "mae_yaw_norm": float(mae[2].item()),
        "band_acc": float(band_acc.item()),
        "ready_f1": float(f1.item()),
        "false_ready_rate": float(false_ready.item()),
        "uncertainty_false_ready": float(uncertain_false_ready.item()),
        "pred_release_band_rate": float(torch.mean(pred_release_band).item()),
        "teacher_ready_xy_in_band_rate": float(teacher_ready_xy_in_band.item()),
        "teacher_ready_z_in_band_rate": float(teacher_ready_z_in_band.item()),
        "teacher_ready_yaw_in_band_rate": float(teacher_ready_yaw_in_band.item()),
        "teacher_ready_ready_prob_pos_rate": float(teacher_ready_prob_pos.item()),
        "teacher_ready_ready_prob_mean": float(teacher_ready_prob_mean.item()),
        "teacher_ready_uncertainty_mean": float(teacher_ready_uncertainty_mean.item()),
        "teacher_ready_rows": float(teacher_ready_rows),
        "teacher_ready_push_rows": float(torch.sum(teacher_ready_mask.float()).item()),
        "close_ready_crossing_rows": float(torch.sum(close_ready_crossing_mask.float()).item()),
        "z_aware_ready_crossing_rows": float(torch.sum(z_aware_ready_crossing_mask.float()).item()),
        "late_ready_logit_lift_rows": float(torch.sum(late_ready_logit_lift_mask.float()).item()),
        "ready_neighborhood_consistency_rows": float(torch.sum(ready_neighborhood_mask.float()).item()),
        "far_negative_rows": float(torch.sum(far_negative_mask.float()).item()),
        "far_negative_ready_prob_mean": float(torch.mean(ready_prob[far_negative_mask]).item()) if torch.any(far_negative_mask) else 0.0,
        "current_profile_hard_negative_veto_rows": float(torch.sum(current_profile_hard_negative_mask.float()).item()),
        "current_profile_hard_negative_ready_prob_mean": float(torch.mean(ready_prob[current_profile_hard_negative_mask]).item()) if torch.any(current_profile_hard_negative_mask) else 0.0,
        "current_profile_hard_negative_pred_yaw_norm_mean": float(torch.mean(pred_metrics[current_profile_hard_negative_mask, 2]).item()) if torch.any(current_profile_hard_negative_mask) else 0.0,
        "very_near_rows": float(very_near_diag["rows"]),
        "very_near_xy_in_band_rate": float(very_near_diag["xy_in_band_rate"]),
        "very_near_z_in_band_rate": float(very_near_diag["z_in_band_rate"]),
        "very_near_yaw_in_band_rate": float(very_near_diag["yaw_in_band_rate"]),
        "very_near_release_band_rate": float(very_near_diag["release_band_rate"]),
        "very_near_ready_prob_mean": float(very_near_diag["ready_prob_mean"]),
        "very_near_ready_prob_pos_rate": float(very_near_diag["ready_prob_pos_rate"]),
        "ready_support_rows": float(ready_support_diag["rows"]),
        "ready_support_xy_in_band_rate": float(ready_support_diag["xy_in_band_rate"]),
        "ready_support_z_in_band_rate": float(ready_support_diag["z_in_band_rate"]),
        "ready_support_yaw_in_band_rate": float(ready_support_diag["yaw_in_band_rate"]),
        "ready_support_release_band_rate": float(ready_support_diag["release_band_rate"]),
        "ready_support_ready_prob_mean": float(ready_support_diag["ready_prob_mean"]),
        "ready_support_ready_prob_pos_rate": float(ready_support_diag["ready_prob_pos_rate"]),
        "ready_support_uncertainty_mean": float(ready_support_diag["uncertainty_mean"]),
    }


@torch.no_grad()
def evaluate(
    model,
    loader,
    device,
    lambda_band: float,
    lambda_ready: float,
    lambda_uncertainty: float,
    lambda_xy: float,
    lambda_z: float,
    lambda_yaw: float,
    lambda_teacher_ready_push: float = 0.0,
    lambda_far_negative_calib: float = 0.0,
    lambda_close_ready_crossing: float = 0.0,
    lambda_z_aware_ready_crossing: float = 0.0,
    lambda_late_ready_logit_lift: float = 0.0,
    lambda_ready_neighborhood_consistency: float = 0.0,
    lambda_far_negative_hard: float = 0.0,
    far_negative_min: float = 2.5,
    crossing_xy_max: float = 1.05,
    crossing_z_max: float = 1.0,
    crossing_yaw_max: float = 1.0,
    z_ready_xy_near_max: float = 1.10,
    z_ready_z_max: float = 0.90,
    z_ready_yaw_max: float = 1.00,
    late_ready_xy_max: float = 1.00,
    late_ready_z_max: float = 1.00,
    late_ready_yaw_max: float = 1.00,
    late_ready_uncertainty_max: float = 0.50,
    late_ready_prob_min: float = 0.40,
    late_ready_prob_max: float = 0.50,
    ready_pos_margin_logit: float = 1.0,
    ready_neg_margin_logit: float = 1.0,
    ready_neighborhood_target_logit: float = 0.0,
    consistency_model=None,
    lambda_consistency_band: float = 0.0,
    lambda_consistency_ready: float = 0.0,
    consistency_sources: tuple[str, ...] = (),
    lambda_current_profile_hard_negative_veto: float = 0.0,
    hard_negative_yaw_margin_norm: float = 1.15,
):
    model.eval()
    agg = {"loss": 0.0}
    counts = {"loss_rows": 0.0}
    subset_prefixes = [
        "",
        "very_near_",
        "ready_support_",
        "subset_broad_xy_recovery_",
        "subset_near_xy_hard_",
        "subset_yaw_needed_",
        "subset_far_negative_",
    ]
    diag_metric_keys = [
        "mae_xy_norm",
        "mae_z_norm",
        "mae_yaw_norm",
        "false_ready_rate",
        "xy_in_band_rate",
        "z_in_band_rate",
        "yaw_in_band_rate",
        "release_band_rate",
        "ready_prob_mean",
        "ready_prob_pos_rate",
        "uncertainty_mean",
        "teacher_ready_rate",
        "teacher_ready_xy_in_band_rate",
        "teacher_ready_z_in_band_rate",
        "teacher_ready_yaw_in_band_rate",
        "teacher_ready_ready_prob_mean",
        "band_gt1_rate",
    ]

    def accumulate(prefix: str, diag: dict):
        rows = float(diag["rows"])
        if rows <= 0.0:
            return
        counts[f"{prefix}rows"] = counts.get(f"{prefix}rows", 0.0) + rows
        teacher_ready_rows = float(diag.get("teacher_ready_rows", 0.0))
        counts[f"{prefix}teacher_ready_rows"] = counts.get(f"{prefix}teacher_ready_rows", 0.0) + teacher_ready_rows
        for key, value in diag.items():
            if key in {"rows", "teacher_ready_rows"}:
                continue
            weight = teacher_ready_rows if key.startswith("teacher_ready_") else rows
            if weight <= 0.0:
                continue
            agg[f"{prefix}{key}"] = agg.get(f"{prefix}{key}", 0.0) + float(value) * weight

    for batch in loader:
        loss, metrics = compute_loss(
            model,
            batch,
            device,
            lambda_band,
            lambda_ready,
            lambda_uncertainty,
            lambda_xy,
            lambda_z,
            lambda_yaw,
            lambda_teacher_ready_push,
            lambda_far_negative_calib,
            lambda_close_ready_crossing,
            lambda_z_aware_ready_crossing,
            lambda_late_ready_logit_lift,
            lambda_ready_neighborhood_consistency,
            lambda_far_negative_hard,
            far_negative_min,
            crossing_xy_max,
            crossing_z_max,
            crossing_yaw_max,
            z_ready_xy_near_max,
            z_ready_z_max,
            z_ready_yaw_max,
            late_ready_xy_max,
            late_ready_z_max,
            late_ready_yaw_max,
            late_ready_uncertainty_max,
            late_ready_prob_min,
            late_ready_prob_max,
            ready_pos_margin_logit,
            ready_neg_margin_logit,
            ready_neighborhood_target_logit,
            consistency_model,
            lambda_consistency_band,
            lambda_consistency_ready,
            consistency_sources,
            lambda_current_profile_hard_negative_veto,
            hard_negative_yaw_margin_norm,
        )
        bsz = batch["teacher_truth_handoff_ready"].shape[0]
        agg["loss"] += float(loss.item()) * bsz
        counts["loss_rows"] += bsz

        out = forward_model(model, batch, device)
        pred_metrics = out["pred_metrics_norm"]
        ready_prob = torch.sigmoid(out["ready_logit"])
        uncertainty = out["uncertainty"]
        teacher_metrics = batch["teacher_metrics_norm"].to(device=device, dtype=torch.float32)
        teacher_ready = batch["teacher_truth_handoff_ready"].to(device=device, dtype=torch.float32)
        teacher_band = batch["teacher_band_label"].to(device=device)
        near_xy_hard = batch["near_xy_hard"].to(device=device, dtype=torch.float32) > 0.5
        broad_xy_recovery = batch["broad_xy_recovery"].to(device=device, dtype=torch.float32) > 0.5
        yaw_needed_v1 = batch.get("yaw_needed_v1")
        if yaw_needed_v1 is None:
            yaw_needed_v1 = batch["near_yaw_hard"]
        yaw_needed_v1 = yaw_needed_v1.to(device=device, dtype=torch.float32) > 0.5
        ready_support = batch["ready_support"].to(device=device, dtype=torch.float32) > 0.5
        far_negative = (
            (teacher_ready <= 0.5)
            & (teacher_band == 0)
            & torch.any(teacher_metrics >= float(far_negative_min), dim=-1)
        )

        accumulate(
            "",
            _subset_diag(
                pred_metrics,
                teacher_metrics,
                ready_prob,
                uncertainty,
                teacher_ready,
                teacher_band,
                torch.ones_like(teacher_ready, dtype=torch.bool),
            ),
        )
        accumulate(
            "very_near_",
            _subset_diag(
                pred_metrics,
                teacher_metrics,
                ready_prob,
                uncertainty,
                teacher_ready,
                teacher_band,
                teacher_band > 0,
            ),
        )
        accumulate(
            "ready_support_",
            _subset_diag(
                pred_metrics,
                teacher_metrics,
                ready_prob,
                uncertainty,
                teacher_ready,
                teacher_band,
                ready_support,
            ),
        )
        accumulate(
            "subset_broad_xy_recovery_",
            _subset_diag(
                pred_metrics,
                teacher_metrics,
                ready_prob,
                uncertainty,
                teacher_ready,
                teacher_band,
                broad_xy_recovery,
            ),
        )
        accumulate(
            "subset_near_xy_hard_",
            _subset_diag(
                pred_metrics,
                teacher_metrics,
                ready_prob,
                uncertainty,
                teacher_ready,
                teacher_band,
                near_xy_hard,
            ),
        )
        accumulate(
            "subset_yaw_needed_",
            _subset_diag(
                pred_metrics,
                teacher_metrics,
                ready_prob,
                uncertainty,
                teacher_ready,
                teacher_band,
                yaw_needed_v1,
            ),
        )
        accumulate(
            "subset_far_negative_",
            _subset_diag(
                pred_metrics,
                teacher_metrics,
                ready_prob,
                uncertainty,
                teacher_ready,
                teacher_band,
                far_negative,
            ),
        )

        source_names = batch["source_name"]
        for source_name in sorted(set(source_names)):
            source_mask = torch.tensor([name == source_name for name in source_names], device=device, dtype=torch.bool)
            accumulate(
                f"source_{source_name}_",
                _subset_diag(
                    pred_metrics,
                    teacher_metrics,
                    ready_prob,
                    uncertainty,
                    teacher_ready,
                    teacher_band,
                    source_mask,
                ),
            )

    out_metrics = {"loss": agg["loss"] / max(counts["loss_rows"], 1.0)}
    for key, value in agg.items():
        if key == "loss":
            continue
        prefix = ""
        base_key = key
        if "_" in key:
            parts = key.split("_")
        teacher_ready_weighted = key.endswith("teacher_ready_xy_in_band_rate") or key.endswith("teacher_ready_z_in_band_rate") or key.endswith("teacher_ready_yaw_in_band_rate") or key.endswith("teacher_ready_ready_prob_mean")
        prefix_candidates = [
            "very_near_",
            "ready_support_",
            "subset_broad_xy_recovery_",
            "subset_near_xy_hard_",
        ]
        prefix_candidates += [k[:-4] for k in counts.keys() if k.startswith("source_") and k.endswith("rows")]
        for candidate_prefix in prefix_candidates:
            if key.startswith(candidate_prefix):
                prefix = candidate_prefix
                base_key = key[len(candidate_prefix):]
                break
        row_count_key = f"{prefix}teacher_ready_rows" if teacher_ready_weighted else f"{prefix}rows"
        denom = counts.get(row_count_key, 0.0)
        out_metrics[key] = value / max(denom, 1.0)
    for key, value in counts.items():
        if key == "loss_rows":
            continue
        out_metrics[key] = float(value)

    # Keep sparse validation splits stable: even if a subset has zero rows,
    # downstream checkpoint selection should still see explicit zero-valued keys.
    for prefix in subset_prefixes:
        out_metrics.setdefault(f"{prefix}rows", 0.0)
        out_metrics.setdefault(f"{prefix}teacher_ready_rows", 0.0)
        for key in diag_metric_keys:
            out_metrics.setdefault(f"{prefix}{key}", 0.0)
    return out_metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_npz", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--val_ratio", type=float, default=0.2)
    ap.add_argument(
        "--val_episode_csv",
        type=str,
        default="",
        help="Optional fixed validation episodes, comma-separated. Overrides --val_ratio.",
    )
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--lambda_band", type=float, default=0.5)
    ap.add_argument("--lambda_ready", type=float, default=0.25)
    ap.add_argument("--lambda_uncertainty", type=float, default=0.0)
    ap.add_argument("--lambda_xy", type=float, default=1.5)
    ap.add_argument("--lambda_z", type=float, default=0.5)
    ap.add_argument("--lambda_yaw", type=float, default=1.0)
    ap.add_argument("--lambda_teacher_ready_push", type=float, default=0.0)
    ap.add_argument("--lambda_far_negative_calib", type=float, default=0.0)
    ap.add_argument("--lambda_close_ready_crossing", type=float, default=0.0)
    ap.add_argument("--lambda_z_aware_ready_crossing", type=float, default=0.0)
    ap.add_argument("--lambda_late_ready_logit_lift", type=float, default=0.0)
    ap.add_argument("--lambda_ready_neighborhood_consistency", type=float, default=0.0)
    ap.add_argument("--lambda_far_negative_hard", type=float, default=0.0)
    ap.add_argument("--far_negative_min", type=float, default=2.5)
    ap.add_argument("--crossing_xy_max", type=float, default=1.05)
    ap.add_argument("--crossing_z_max", type=float, default=1.0)
    ap.add_argument("--crossing_yaw_max", type=float, default=1.0)
    ap.add_argument("--z_ready_xy_near_max", type=float, default=1.10)
    ap.add_argument("--z_ready_z_max", type=float, default=0.90)
    ap.add_argument("--z_ready_yaw_max", type=float, default=1.00)
    ap.add_argument("--late_ready_xy_max", type=float, default=1.00)
    ap.add_argument("--late_ready_z_max", type=float, default=1.00)
    ap.add_argument("--late_ready_yaw_max", type=float, default=1.00)
    ap.add_argument("--late_ready_uncertainty_max", type=float, default=0.50)
    ap.add_argument("--late_ready_prob_min", type=float, default=0.40)
    ap.add_argument("--late_ready_prob_max", type=float, default=0.50)
    ap.add_argument("--ready_pos_margin_logit", type=float, default=1.0)
    ap.add_argument("--ready_neg_margin_logit", type=float, default=1.0)
    ap.add_argument("--ready_neighborhood_target_logit", type=float, default=0.0)
    ap.add_argument("--lambda_consistency_band", type=float, default=0.0)
    ap.add_argument("--lambda_consistency_ready", type=float, default=0.0)
    ap.add_argument("--lambda_current_profile_hard_negative_veto", type=float, default=0.0)
    ap.add_argument("--hard_negative_yaw_margin_norm", type=float, default=1.15)
    ap.add_argument("--consistency_ckpt", type=str, default="")
    ap.add_argument("--consistency_source", action="append", default=[])
    ap.add_argument("--weighted_sampling", action="store_true")
    ap.add_argument("--sampler_weight_power", type=float, default=1.0)
    ap.add_argument("--deploy_false_ready_max", type=float, default=1e-8)
    ap.add_argument(
        "--init_ckpt",
        type=str,
        default="",
        help="Optional checkpoint to warm-start student_handoff_state_head_v2 from.",
    )
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    dataset = HandoffStateDatasetV2(args.dataset_npz)
    fixed_val_eps = [int(x) for x in args.val_episode_csv.split(",") if str(x).strip()] if args.val_episode_csv else []
    fixed_val_present = []
    if fixed_val_eps:
        train_idx, val_idx, fixed_val_present = split_by_fixed_val_episodes(dataset, fixed_val_eps)
    else:
        train_idx, val_idx = split_by_episode(dataset, args.val_ratio, args.seed)
    train_set = Subset(dataset, train_idx)
    val_set = Subset(dataset, val_idx)
    if args.weighted_sampling:
        raw_w = np.asarray(dataset.data["sample_weight"][train_idx], dtype=np.float64)
        raw_w = np.clip(raw_w, 1e-6, None) ** float(args.sampler_weight_power)
        sampler = WeightedRandomSampler(
            weights=torch.as_tensor(raw_w, dtype=torch.double),
            num_samples=len(train_idx),
            replacement=True,
        )
        train_loader = DataLoader(train_set, batch_size=args.batch_size, sampler=sampler, shuffle=False, num_workers=0)
    else:
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = StudentHandoffStateHeadV2().to(device)
    init_ckpt_path = args.init_ckpt.strip()
    init_ckpt_loaded = ""
    if init_ckpt_path:
        ckpt = torch.load(init_ckpt_path, map_location="cpu")
        state = ckpt.get("model_state_dict", ckpt)
        missing, unexpected = model.load_state_dict(state, strict=False)
        init_ckpt_loaded = init_ckpt_path
        print(
            json.dumps(
                {
                    "event": "loaded_init_ckpt",
                    "path": init_ckpt_path,
                    "missing_keys": sorted(list(missing)),
                    "unexpected_keys": sorted(list(unexpected)),
                }
            )
        )
    consistency_ckpt_path = args.consistency_ckpt.strip()
    consistency_model = None
    consistency_sources = tuple(args.consistency_source)
    if consistency_ckpt_path:
        consistency_model = StudentHandoffStateHeadV2().to(device)
        ckpt = torch.load(consistency_ckpt_path, map_location="cpu")
        state = ckpt.get("model_state_dict", ckpt)
        missing, unexpected = consistency_model.load_state_dict(state, strict=False)
        consistency_model.eval()
        for p in consistency_model.parameters():
            p.requires_grad_(False)
        print(
            json.dumps(
                {
                    "event": "loaded_consistency_ckpt",
                    "path": consistency_ckpt_path,
                    "missing_keys": sorted(list(missing)),
                    "unexpected_keys": sorted(list(unexpected)),
                    "sources": list(consistency_sources),
                }
            )
        )
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    split_info = {
        "train_rows": int(len(train_idx)),
        "val_rows": int(len(val_idx)),
        "train_episodes": int(np.unique(dataset.data["episode_index"][train_idx]).size),
        "val_episodes": int(np.unique(dataset.data["episode_index"][val_idx]).size),
        "fixed_val_episode_csv": args.val_episode_csv,
        "fixed_val_episode_indices": fixed_val_present,
        "init_ckpt": init_ckpt_loaded,
        "consistency_ckpt": consistency_ckpt_path,
        "consistency_sources": list(consistency_sources),
    }
    (output_dir / "split_info.json").write_text(json.dumps(split_info, indent=2))

    def _save_ckpt(path: Path, val_metrics: dict, history_rows: list[dict]) -> None:
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "dataset_npz": str(args.dataset_npz),
                "history": history_rows,
                "val_metrics": val_metrics,
                "model_kind": "student_handoff_state_head_v2",
            },
            path,
        )

    best_key = float("inf")
    best_geom_key = float("inf")
    best_anchor_key = None
    best_ready_key = None
    best_phasea_deploy_key = None
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss_total = 0.0
        train_count = 0
        for batch in train_loader:
            loss, _ = compute_loss(
                model,
                batch,
                device,
                args.lambda_band,
                args.lambda_ready,
                args.lambda_uncertainty,
                args.lambda_xy,
                args.lambda_z,
                args.lambda_yaw,
                args.lambda_teacher_ready_push,
                args.lambda_far_negative_calib,
                args.lambda_close_ready_crossing,
                args.lambda_z_aware_ready_crossing,
                args.lambda_late_ready_logit_lift,
                args.lambda_ready_neighborhood_consistency,
                args.lambda_far_negative_hard,
                args.far_negative_min,
                args.crossing_xy_max,
                args.crossing_z_max,
                args.crossing_yaw_max,
                args.z_ready_xy_near_max,
                args.z_ready_z_max,
                args.z_ready_yaw_max,
                args.late_ready_xy_max,
                args.late_ready_z_max,
                args.late_ready_yaw_max,
                args.late_ready_uncertainty_max,
                args.late_ready_prob_min,
                args.late_ready_prob_max,
                args.ready_pos_margin_logit,
                args.ready_neg_margin_logit,
                args.ready_neighborhood_target_logit,
                consistency_model,
                args.lambda_consistency_band,
                args.lambda_consistency_ready,
                consistency_sources,
                args.lambda_current_profile_hard_negative_veto,
                args.hard_negative_yaw_margin_norm,
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            bsz = batch["teacher_truth_handoff_ready"].shape[0]
            train_loss_total += float(loss.item()) * bsz
            train_count += bsz
        train_loss = train_loss_total / max(train_count, 1)
        val_metrics = evaluate(
            model,
            val_loader,
            device,
            args.lambda_band,
            args.lambda_ready,
            args.lambda_uncertainty,
            args.lambda_xy,
            args.lambda_z,
            args.lambda_yaw,
            args.lambda_teacher_ready_push,
            args.lambda_far_negative_calib,
            args.lambda_close_ready_crossing,
            args.lambda_z_aware_ready_crossing,
            args.lambda_late_ready_logit_lift,
            args.lambda_ready_neighborhood_consistency,
            args.lambda_far_negative_hard,
            args.far_negative_min,
            args.crossing_xy_max,
            args.crossing_z_max,
            args.crossing_yaw_max,
            args.z_ready_xy_near_max,
            args.z_ready_z_max,
            args.z_ready_yaw_max,
            args.late_ready_xy_max,
            args.late_ready_z_max,
            args.late_ready_yaw_max,
            args.late_ready_uncertainty_max,
            args.late_ready_prob_min,
            args.late_ready_prob_max,
            args.ready_pos_margin_logit,
            args.ready_neg_margin_logit,
            args.ready_neighborhood_target_logit,
            consistency_model,
            args.lambda_consistency_band,
            args.lambda_consistency_ready,
            consistency_sources,
            args.lambda_current_profile_hard_negative_veto,
            args.hard_negative_yaw_margin_norm,
        )
        row = {"epoch": epoch, "train_loss": train_loss, **val_metrics}
        history.append(row)
        print(json.dumps(row))
        score = (
            val_metrics["loss"]
            + 0.25 * (1.0 - val_metrics["very_near_xy_in_band_rate"])
            + 0.25 * (1.0 - val_metrics["very_near_yaw_in_band_rate"])
            + 0.50 * (1.0 - val_metrics["ready_support_release_band_rate"])
            + 0.25 * (1.0 - val_metrics["ready_support_ready_prob_mean"])
        )
        if score < best_key:
            best_key = score
            _save_ckpt(output_dir / "student_handoff_state_head_v2_best.pt", val_metrics, history)

        geom_score = (
            val_metrics["loss"]
            + 0.35 * (1.0 - val_metrics["teacher_ready_xy_in_band_rate"])
            + 0.25 * (1.0 - val_metrics["ready_support_xy_in_band_rate"])
            + 0.20 * (1.0 - val_metrics["very_near_xy_in_band_rate"])
            + 5.0 * val_metrics["false_ready_rate"]
        )
        if geom_score < best_geom_key:
            best_geom_key = geom_score
            _save_ckpt(output_dir / "student_handoff_state_head_v2_best_geom.pt", val_metrics, history)

        anchor_key = (
            -float(val_metrics["false_ready_rate"] <= 1e-8),
            -val_metrics["teacher_ready_xy_in_band_rate"],
            -val_metrics["ready_support_xy_in_band_rate"],
            -val_metrics["teacher_ready_ready_prob_mean"],
            val_metrics["loss"],
        )
        if best_anchor_key is None or anchor_key < best_anchor_key:
            best_anchor_key = anchor_key
            _save_ckpt(output_dir / "student_handoff_state_head_v2_best_anchor.pt", val_metrics, history)

        ready_key = (
            -float(val_metrics["false_ready_rate"] <= 1e-8),
            -val_metrics["teacher_ready_ready_prob_mean"],
            -val_metrics["ready_support_release_band_rate"],
            -val_metrics["ready_support_xy_in_band_rate"],
            val_metrics["loss"],
        )
        if best_ready_key is None or ready_key < best_ready_key:
            best_ready_key = ready_key
            _save_ckpt(output_dir / "student_handoff_state_head_v2_best_ready.pt", val_metrics, history)

        phasea_deploy_key = (
            -float(val_metrics["false_ready_rate"] <= args.deploy_false_ready_max),
            -float(val_metrics["ready_support_release_band_rate"] > 1e-8),
            -(
                0.35 * val_metrics["teacher_ready_xy_in_band_rate"]
                + 0.25 * val_metrics["ready_support_xy_in_band_rate"]
                + 0.25 * val_metrics["ready_support_release_band_rate"]
                + 0.15 * val_metrics["teacher_ready_ready_prob_mean"]
            ),
            val_metrics["loss"],
        )
        if best_phasea_deploy_key is None or phasea_deploy_key < best_phasea_deploy_key:
            best_phasea_deploy_key = phasea_deploy_key
            _save_ckpt(output_dir / "student_handoff_state_head_v2_best_phaseA_deploy.pt", val_metrics, history)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "dataset_npz": str(args.dataset_npz),
            "history": history,
            "val_metrics": history[-1] if history else {},
            "model_kind": "student_handoff_state_head_v2",
        },
        output_dir / "student_handoff_state_head_v2_final.pt",
    )
    (output_dir / "train_history.json").write_text(json.dumps(history, indent=2))


if __name__ == "__main__":
    main()
