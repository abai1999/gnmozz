"""Learned contract-aware local geometry localizer for Coarse2Contact v2."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F

from .localizers import LocalGeometryError
from prismatic.robot.residual_transforms import world_delta_to_local


def build_vocab(values: Iterable[str], *, add_unknown: bool = True) -> dict[str, int]:
    vocab: dict[str, int] = {}
    if add_unknown:
        vocab["<unk>"] = 0
    for value in values:
        key = str(value)
        if key not in vocab:
            vocab[key] = len(vocab)
    return vocab


def _lookup(vocab: Mapping[str, int], value: str) -> int:
    if value in vocab:
        return int(vocab[value])
    return int(vocab.get("<unk>", 0))


def encode_dofs(controlled_dofs: Sequence[str], vocab: Mapping[str, int]) -> torch.Tensor:
    vec = torch.zeros(len(vocab), dtype=torch.float32)
    for dof in controlled_dofs:
        idx = vocab.get(str(dof))
        if idx is not None:
            vec[int(idx)] = 1.0
    return vec


class _ImageEncoder(nn.Module):
    def __init__(self, in_channels: int = 6, hidden_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 96, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, hidden_dim, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, *, pooled: bool = True) -> torch.Tensor:
        feat = self.net(x)
        if pooled:
            return F.adaptive_avg_pool2d(feat, output_size=1).flatten(1)
        return feat


def _center_crop_box(width: int, height: int, crop_size: int) -> tuple[int, int, int, int]:
    crop = int(max(8, min(crop_size, min(width, height))))
    half = crop // 2
    cx = width // 2
    cy = height // 2
    x0 = max(0, cx - half)
    y0 = max(0, cy - half)
    x1 = min(width, x0 + crop)
    y1 = min(height, y0 + crop)
    if x1 - x0 < crop:
        x0 = max(0, x1 - crop)
    if y1 - y0 < crop:
        y0 = max(0, y1 - crop)
    return int(x0), int(y0), int(x1), int(y1)


def _roi_center_from_prior(observation: Any, robot_state: Any, width: int, height: int) -> tuple[float, float]:
    cx = 0.5 * (width - 1)
    cy = 0.5 * (height - 1)
    planner_delta = None
    if isinstance(robot_state, Mapping):
        planner_delta = robot_state.get("planner_delta_7d", None)
    if planner_delta is None:
        return cx, cy
    try:
        delta = np.asarray(planner_delta, dtype=np.float32).reshape(-1)[:6]
        gripper_pose = _obs_get(observation, "gripper_pose", None)
        quat = np.asarray(gripper_pose, dtype=np.float32).reshape(-1)[3:7] if gripper_pose is not None and np.asarray(gripper_pose).size >= 7 else np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        local_delta = world_delta_to_local(delta, quat).astype(np.float32)
        # Heuristic: laterally shift the ROI toward the planner's coarse target prior.
        px_per_meter = 1200.0
        shift_x = float(np.clip(local_delta[0] * px_per_meter, -0.28 * width, 0.28 * width))
        shift_y = float(np.clip(-local_delta[1] * px_per_meter, -0.28 * height, 0.28 * height))
        return cx + shift_x, cy + shift_y
    except Exception:
        return cx, cy


def _resize_rgbd(rgbd: np.ndarray, size: int) -> np.ndarray:
    size = int(max(8, size))
    rgb = np.clip(rgbd[..., :3] * 255.0, 0.0, 255.0).astype(np.uint8)
    depth = np.clip(rgbd[..., 3], 0.0, 1.0)
    rgb_img = Image.fromarray(rgb, mode="RGB").resize((size, size), resample=Image.BILINEAR)
    depth_img = Image.fromarray(depth.astype(np.float32), mode="F").resize((size, size), resample=Image.BILINEAR)
    rgb_arr = np.asarray(rgb_img, dtype=np.float32) / 255.0
    depth_arr = np.asarray(depth_img, dtype=np.float32)
    h, w = depth_arr.shape[:2]
    xs = np.linspace(-1.0, 1.0, num=w, dtype=np.float32)
    ys = np.linspace(-1.0, 1.0, num=h, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(xs, ys)
    coord = np.stack([grid_x, grid_y], axis=-1)
    return np.concatenate([rgb_arr, depth_arr[..., None], coord], axis=-1).astype(np.float32)


def _make_gaussian_heatmap(
    center_uv: tuple[float, float] | np.ndarray,
    *,
    size: int,
    sigma: float = 1.5,
    valid: bool = True,
) -> np.ndarray:
    size = int(max(4, size))
    if not valid:
        return np.zeros((size, size), dtype=np.float32)
    u, v = np.asarray(center_uv, dtype=np.float32).reshape(-1)[:2]
    u = float(np.clip(u, 0.0, 1.0)) * (size - 1)
    v = float(np.clip(v, 0.0, 1.0)) * (size - 1)
    yy, xx = np.meshgrid(np.arange(size, dtype=np.float32), np.arange(size, dtype=np.float32), indexing="ij")
    heat = np.exp(-((xx - u) ** 2 + (yy - v) ** 2) / (2.0 * float(max(sigma, 1e-6)) ** 2))
    return heat.astype(np.float32)


def _make_segment_heatmap(
    start_uv: tuple[float, float] | np.ndarray,
    end_uv: tuple[float, float] | np.ndarray,
    *,
    size: int,
    sigma: float = 1.0,
    valid: bool = True,
) -> np.ndarray:
    size = int(max(4, size))
    if not valid:
        return np.zeros((size, size), dtype=np.float32)
    start = np.asarray(start_uv, dtype=np.float32).reshape(-1)[:2]
    end = np.asarray(end_uv, dtype=np.float32).reshape(-1)[:2]
    start = np.clip(start, 0.0, 1.0) * (size - 1)
    end = np.clip(end, 0.0, 1.0) * (size - 1)
    yy, xx = np.meshgrid(np.arange(size, dtype=np.float32), np.arange(size, dtype=np.float32), indexing="ij")
    vx = float(end[0] - start[0])
    vy = float(end[1] - start[1])
    denom = float(vx * vx + vy * vy)
    if denom <= 1e-8:
        return _make_gaussian_heatmap(start_uv, size=size, sigma=sigma, valid=valid)
    t = ((xx - start[0]) * vx + (yy - start[1]) * vy) / denom
    t = np.clip(t, 0.0, 1.0)
    proj_x = start[0] + t * vx
    proj_y = start[1] + t * vy
    dist2 = (xx - proj_x) ** 2 + (yy - proj_y) ** 2
    heat = np.exp(-dist2 / (2.0 * float(max(sigma, 1e-6)) ** 2))
    taper = np.exp(-((t - 0.5) ** 2) / 0.08)
    return (heat * (0.5 + 0.5 * taper)).astype(np.float32)


def _normalize_2d(vec: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return vec / torch.clamp(torch.linalg.norm(vec, dim=-1, keepdim=True), min=eps)


def _softargmax_2d(logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if logits.ndim != 4:
        raise ValueError(f"Expected logits in BxCxHxW, got shape {tuple(logits.shape)}")
    probs = torch.softmax(logits.flatten(2), dim=-1).view_as(logits)
    grid_y, grid_x = torch.meshgrid(
        torch.linspace(0.0, 1.0, logits.shape[-2], device=logits.device, dtype=logits.dtype),
        torch.linspace(0.0, 1.0, logits.shape[-1], device=logits.device, dtype=logits.dtype),
        indexing="ij",
    )
    grid_x = grid_x.unsqueeze(0).unsqueeze(0)
    grid_y = grid_y.unsqueeze(0).unsqueeze(0)
    exp_x = torch.sum(probs * grid_x, dim=(-2, -1))
    exp_y = torch.sum(probs * grid_y, dim=(-2, -1))
    peak = torch.amax(probs, dim=(-2, -1))
    return exp_x, exp_y, peak


class DepthGeometryLocalizerNet(nn.Module):
    """Tiny contract-aware RGBD localizer that predicts local correction deltas."""

    def __init__(
        self,
        *,
        skill_type_vocab: Mapping[str, int],
        stage_vocab: Mapping[str, int],
        entity_vocab: Mapping[str, int],
        dof_vocab: Mapping[str, int],
        image_in_channels: int = 6,
        image_hidden_dim: int = 128,
        embed_dim: int = 24,
        output_dim: int = 5,
        prediction_mode: str = "regression",
        heatmap_size: int = 16,
        heatmap_sigma: float = 1.5,
        heatmap_xy_range_m: float = 0.04,
        heatmap_channels: int = 3,
        heatmap_pos_weight: float = 8.0,
    ) -> None:
        super().__init__()
        self.skill_type_vocab = dict(skill_type_vocab)
        self.stage_vocab = dict(stage_vocab)
        self.entity_vocab = dict(entity_vocab)
        self.dof_vocab = dict(dof_vocab)
        self.image_in_channels = int(image_in_channels)
        self.prediction_mode = str(prediction_mode)
        self.heatmap_size = int(heatmap_size)
        self.heatmap_sigma = float(heatmap_sigma)
        self.heatmap_xy_range_m = float(heatmap_xy_range_m)
        self.heatmap_channels = int(heatmap_channels)
        self.heatmap_pos_weight = float(heatmap_pos_weight)
        self.image_encoder = _ImageEncoder(in_channels=self.image_in_channels, hidden_dim=image_hidden_dim)
        self.skill_emb = nn.Embedding(max(len(self.skill_type_vocab), 1), embed_dim)
        self.stage_emb = nn.Embedding(max(len(self.stage_vocab), 1), embed_dim)
        self.entity_emb = nn.Embedding(max(len(self.entity_vocab), 1), embed_dim)
        self.contract_mlp = nn.Sequential(
            nn.Linear(embed_dim * 4 + len(self.dof_vocab), 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 96),
            nn.ReLU(inplace=True),
        )
        if self.prediction_mode == "heatmap":
            self.contract_to_spatial = nn.Linear(96, 32)
            self.heatmap_fusion = nn.Sequential(
                nn.Conv2d(image_hidden_dim + 32, image_hidden_dim, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(image_hidden_dim, max(self.heatmap_channels, 3), kernel_size=1),
            )
            self.frame_aux_head = nn.Sequential(
                nn.Linear(image_hidden_dim + 96, 128),
                nn.ReLU(inplace=True),
                nn.Linear(128, 4),
            )
        else:
            self.head = nn.Sequential(
                nn.Linear(image_hidden_dim + 96, 128),
                nn.ReLU(inplace=True),
                nn.Linear(128, 64),
                nn.ReLU(inplace=True),
                nn.Linear(64, output_dim),
            )

    @classmethod
    def from_vocab(cls, vocab: Mapping[str, Mapping[str, int]], **kwargs) -> "DepthGeometryLocalizerNet":
        return cls(
            skill_type_vocab=vocab["skill_type"],
            stage_vocab=vocab["stage_name"],
            entity_vocab=vocab["entity"],
            dof_vocab=vocab["controlled_dofs"],
            **kwargs,
        )

    def forward(
        self,
        image_rgbd: torch.Tensor,
        skill_type_id: torch.Tensor,
        stage_id: torch.Tensor,
        target_entity_id: torch.Tensor,
        reference_entity_id: torch.Tensor,
        dof_vec: torch.Tensor,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        if self.prediction_mode == "heatmap":
            image_feat = self.image_encoder(image_rgbd, pooled=False)
            if image_feat.shape[-2:] != (self.heatmap_size, self.heatmap_size):
                image_feat = F.interpolate(image_feat, size=(self.heatmap_size, self.heatmap_size), mode="bilinear", align_corners=False)
            pooled_image = F.adaptive_avg_pool2d(image_feat, output_size=1).flatten(1)
        else:
            image_feat = self.image_encoder(image_rgbd, pooled=True)
        contract = torch.cat(
            [
                self.skill_emb(skill_type_id),
                self.stage_emb(stage_id),
                self.entity_emb(target_entity_id),
                self.entity_emb(reference_entity_id),
                dof_vec.float(),
            ],
            dim=-1,
        )
        contract_feat = self.contract_mlp(contract)
        if self.prediction_mode == "heatmap":
            spatial = self.contract_to_spatial(contract_feat).unsqueeze(-1).unsqueeze(-1)
            spatial = spatial.expand(-1, -1, image_feat.shape[-2], image_feat.shape[-1])
            fused = torch.cat([image_feat, spatial], dim=-3)
            logits = self.heatmap_fusion(fused)
            aux = self.frame_aux_head(torch.cat([pooled_image, contract_feat], dim=-1))
            out = {
                "center_heatmap_logits": logits[:, 0:1],
                "dz_pred": aux[:, 0:1],
                "axis_dir_xy": aux[:, 1:3],
                "confidence_logit": aux[:, 3],
            }
            out["axis_pos_heatmap_logits"] = logits[:, 1:2]
            out["axis_neg_heatmap_logits"] = logits[:, 2:3]
            return out
        out = self.head(torch.cat([image_feat, contract_feat], dim=-1))
        return out

    def predict(
        self,
        image_rgbd: torch.Tensor,
        skill_type_id: torch.Tensor,
        stage_id: torch.Tensor,
        target_entity_id: torch.Tensor,
        reference_entity_id: torch.Tensor,
        dof_vec: torch.Tensor,
        *,
        xy_range_m: float | None = None,
    ) -> dict[str, torch.Tensor]:
        pred = self.forward(image_rgbd, skill_type_id, stage_id, target_entity_id, reference_entity_id, dof_vec)
        if self.prediction_mode == "heatmap":
            xy_range_m = float(self.heatmap_xy_range_m if xy_range_m is None else xy_range_m)
            center_logits = pred["center_heatmap_logits"]
            axis_pos_logits = pred["axis_pos_heatmap_logits"]
            axis_neg_logits = pred["axis_neg_heatmap_logits"]
            conf_logit = pred["confidence_logit"]
            center_x, center_y, center_peak = _softargmax_2d(center_logits)
            axis_pos_x, axis_pos_y, axis_pos_peak = _softargmax_2d(axis_pos_logits)
            axis_neg_x, axis_neg_y, axis_neg_peak = _softargmax_2d(axis_neg_logits)
            center_u = center_x[:, 0]
            center_v = center_y[:, 0]
            axis_pos_u = axis_pos_x[:, 0]
            axis_pos_v = axis_pos_y[:, 0]
            axis_neg_u = axis_neg_x[:, 0]
            axis_neg_v = axis_neg_y[:, 0]
            center_dx = (center_u - 0.5) * 2.0 * xy_range_m
            center_dy = (center_v - 0.5) * 2.0 * xy_range_m
            axis_dir = _normalize_2d(pred["axis_dir_xy"])
            dyaw = torch.atan2(axis_dir[:, 1], axis_dir[:, 0])
            dz = pred["dz_pred"][:, 0]
            return {
                "dx": center_dx,
                "dy": center_dy,
                "dz": dz,
                "dyaw": dyaw,
                "confidence": torch.sigmoid(conf_logit),
                "center_u": center_u,
                "center_v": center_v,
                "axis_u": axis_pos_u,
                "axis_v": axis_pos_v,
                "axis_pos_u": axis_pos_u,
                "axis_pos_v": axis_pos_v,
                "axis_neg_u": axis_neg_u,
                "axis_neg_v": axis_neg_v,
                "axis_dir_x": axis_dir[:, 0],
                "axis_dir_y": axis_dir[:, 1],
                "confidence_logit": conf_logit,
                "center_peak": center_peak[:, 0],
                "axis_pos_peak": axis_pos_peak[:, 0],
                "axis_neg_peak": axis_neg_peak[:, 0],
            }
        dx = pred[..., 0]
        dy = pred[..., 1]
        dz = pred[..., 2] if pred.shape[-1] >= 5 else torch.zeros_like(pred[..., 0])
        dyaw = pred[..., 3] if pred.shape[-1] >= 5 else pred[..., 2]
        conf_logit = pred[..., 4] if pred.shape[-1] >= 5 else pred[..., 3]
        return {
            "dx": dx,
            "dy": dy,
            "dz": dz,
            "dyaw": dyaw,
            "confidence": torch.sigmoid(conf_logit),
        }


@dataclass(frozen=True)
class DepthLocalizerPrediction:
    dx: float
    dy: float
    dyaw: float
    confidence: float


@dataclass(frozen=True)
class RingFramePrediction:
    visible: float
    center_u: float
    center_v: float
    axis_pos_u: float
    axis_pos_v: float
    axis_neg_u: float
    axis_neg_v: float
    confidence: float


@dataclass(frozen=True)
class GraspSkillPrediction:
    dx: float
    dy: float
    dz: float
    dyaw: float
    ready_to_close: float
    confidence: float


@dataclass(frozen=True)
class GraspRecoveryPrediction:
    dx: float
    dy: float
    dyaw: float
    confidence: float


def load_ring_frame_localizer_checkpoint(
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> tuple[RingFrameLocalizerNet, dict[str, Any]]:
    ckpt = torch.load(Path(path), map_location=map_location)
    model = RingFrameLocalizerNet(**ckpt.get("config", {}))
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt


class RingFrameLocalizerNet(nn.Module):
    """Object-frame predictor for the ring, separate from control-action imitation."""

    def __init__(self, *, image_in_channels: int = 6, image_hidden_dim: int = 128, heatmap_size: int = 32) -> None:
        super().__init__()
        self.heatmap_size = int(heatmap_size)
        self.image_encoder = _ImageEncoder(in_channels=image_in_channels, hidden_dim=image_hidden_dim)
        self.heatmap_head = nn.Sequential(
            nn.Conv2d(image_hidden_dim, image_hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(image_hidden_dim, 3, kernel_size=1),
        )
        self.conf_head = nn.Sequential(
            nn.Linear(image_hidden_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 2),
        )

    def forward(self, image_rgbd: torch.Tensor) -> dict[str, torch.Tensor]:
        feat = self.image_encoder(image_rgbd, pooled=False)
        if feat.shape[-2:] != (self.heatmap_size, self.heatmap_size):
            feat = F.interpolate(feat, size=(self.heatmap_size, self.heatmap_size), mode="bilinear", align_corners=False)
        logits = self.heatmap_head(feat)
        pooled = F.adaptive_avg_pool2d(feat, output_size=1).flatten(1)
        conf = self.conf_head(pooled)
        return {
            "center_heatmap_logits": logits[:, 0:1],
            "axis_pos_heatmap_logits": logits[:, 1:2],
            "axis_neg_heatmap_logits": logits[:, 2:3],
            "visible_logit": conf[:, 0],
            "confidence_logit": conf[:, 1],
        }


class GraspSkillHeadNet(nn.Module):
    """Pre-close local skill head trained only from successful close-index windows."""

    def __init__(
        self,
        *,
        image_in_channels: int = 6,
        image_hidden_dim: int = 128,
        frame_feature_dim: int = 10,
        proprio_dim: int = 15,
        x_output_scale: float = 0.003,
        y_output_scale: float = 0.003,
    ) -> None:
        super().__init__()
        self.x_output_scale = float(x_output_scale)
        self.y_output_scale = float(y_output_scale)
        self.image_encoder = _ImageEncoder(in_channels=image_in_channels, hidden_dim=image_hidden_dim)
        self.trunk = nn.Sequential(
            nn.Linear(image_hidden_dim + frame_feature_dim + proprio_dim, 160),
            nn.ReLU(inplace=True),
            nn.Linear(160, 96),
            nn.ReLU(inplace=True),
        )
        self.xy_head = nn.Linear(96, 2)
        self.yaw_dir_head = nn.Linear(96, 2)
        self.yaw_observable_head = nn.Linear(96, 1)
        self.z_head = nn.Linear(96, 1)
        self.ready_head = nn.Linear(96, 1)
        self.conf_head = nn.Linear(96, 1)

    def forward(self, image_rgbd: torch.Tensor, frame_features: torch.Tensor, proprio: torch.Tensor) -> dict[str, torch.Tensor]:
        image_feat = self.image_encoder(image_rgbd, pooled=True)
        x = torch.cat([image_feat, frame_features.float(), proprio.float()], dim=-1)
        hidden = self.trunk(x)
        xy_raw = self.xy_head(hidden)
        dx = torch.tanh(xy_raw[:, 0]) * self.x_output_scale
        dy = torch.tanh(xy_raw[:, 1]) * self.y_output_scale
        yaw_dir_raw = self.yaw_dir_head(hidden)
        yaw_dir = F.normalize(yaw_dir_raw, dim=-1, eps=1e-6)
        dyaw = torch.atan2(yaw_dir[:, 1], yaw_dir[:, 0])
        yaw_observable = self.yaw_observable_head(hidden)
        z = self.z_head(hidden)
        ready = self.ready_head(hidden)
        conf = self.conf_head(hidden)
        return {
            "dx": dx,
            "dy": dy,
            "dyaw": dyaw,
            "yaw_dir_x": yaw_dir[:, 0],
            "yaw_dir_y": yaw_dir[:, 1],
            "yaw_observable_logit": yaw_observable[:, 0],
            "dz": z[:, 0],
            "ready_to_close_logit": ready[:, 0],
            "confidence_logit": conf[:, 0],
        }


class GraspRecoveryHeadNet(nn.Module):
    """Recovery head that conditions on planner prior and near-grasp context."""

    def __init__(
        self,
        *,
        image_in_channels: int = 6,
        image_hidden_dim: int = 128,
        frame_feature_dim: int = 10,
        proprio_dim: int = 15,
        planner_prior_dim: int = 6,
        x_output_scale: float = 0.02,
        y_output_scale: float = 0.02,
        yaw_output_scale: float = 0.25,
    ) -> None:
        super().__init__()
        self.x_output_scale = float(x_output_scale)
        self.y_output_scale = float(y_output_scale)
        self.yaw_output_scale = float(yaw_output_scale)
        self.planner_prior_dim = int(planner_prior_dim)
        self.image_encoder = _ImageEncoder(in_channels=image_in_channels, hidden_dim=image_hidden_dim)
        self.prior_mlp = nn.Sequential(
            nn.Linear(max(self.planner_prior_dim, 1), 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 32),
            nn.ReLU(inplace=True),
        )
        self.trunk = nn.Sequential(
            nn.Linear(image_hidden_dim + frame_feature_dim + proprio_dim + 32, 192),
            nn.ReLU(inplace=True),
            nn.Linear(192, 128),
            nn.ReLU(inplace=True),
        )
        self.xy_head = nn.Linear(128, 2)
        self.yaw_head = nn.Linear(128, 1)
        self.conf_head = nn.Linear(128, 1)

    def forward(
        self,
        image_rgbd: torch.Tensor,
        frame_features: torch.Tensor,
        proprio: torch.Tensor,
        planner_prior_delta: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        image_feat = self.image_encoder(image_rgbd, pooled=True)
        prior = planner_prior_delta.float()
        if prior.ndim == 1:
            prior = prior.unsqueeze(0)
        if prior.shape[-1] < self.planner_prior_dim:
            prior = F.pad(prior, (0, self.planner_prior_dim - prior.shape[-1]))
        prior_feat = self.prior_mlp(prior[..., : self.planner_prior_dim])
        x = torch.cat([image_feat, frame_features.float(), proprio.float(), prior_feat], dim=-1)
        hidden = self.trunk(x)
        xy_raw = self.xy_head(hidden)
        dx = torch.tanh(xy_raw[:, 0]) * self.x_output_scale
        dy = torch.tanh(xy_raw[:, 1]) * self.y_output_scale
        dyaw = torch.tanh(self.yaw_head(hidden)[:, 0]) * self.yaw_output_scale
        conf = self.conf_head(hidden)[:, 0]
        return {
            "dx": dx,
            "dy": dy,
            "dyaw": dyaw,
            "confidence_logit": conf,
            "confidence": torch.sigmoid(conf),
        }


def load_grasp_skill_head_checkpoint(path: str | Path, *, map_location: str | torch.device = "cpu") -> tuple[GraspSkillHeadNet, dict]:
    ckpt = torch.load(Path(path), map_location=map_location)
    model = GraspSkillHeadNet(**ckpt.get("config", {}))
    state_dict = dict(ckpt["model_state_dict"])
    if "yaw_observable_head.weight" not in state_dict and "conf_head.weight" in state_dict:
        state_dict["yaw_observable_head.weight"] = state_dict["conf_head.weight"].clone()
    if "yaw_observable_head.bias" not in state_dict and "conf_head.bias" in state_dict:
        state_dict["yaw_observable_head.bias"] = state_dict["conf_head.bias"].clone()
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model, ckpt


def load_grasp_recovery_checkpoint(path: str | Path, *, map_location: str | torch.device = "cpu") -> tuple[GraspRecoveryHeadNet, dict]:
    ckpt = torch.load(Path(path), map_location=map_location)
    config = dict(ckpt.get("config", {}))
    allowed = {
        "image_in_channels",
        "image_hidden_dim",
        "frame_feature_dim",
        "proprio_dim",
        "planner_prior_dim",
        "x_output_scale",
        "y_output_scale",
        "yaw_output_scale",
    }
    config = {k: v for k, v in config.items() if k in allowed}
    model = GraspRecoveryHeadNet(**config)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()
    return model, ckpt


def load_depth_localizer_checkpoint(path: str | Path, *, map_location: str | torch.device = "cpu") -> tuple[DepthGeometryLocalizerNet, dict[str, dict[str, int]]]:
    ckpt = torch.load(Path(path), map_location=map_location)
    vocab = ckpt["vocab"]
    config = dict(ckpt.get("config", {}))
    state_dict = ckpt["model_state_dict"]
    if "prediction_mode" not in config:
        if any(key.startswith("heatmap_fusion.") or key.startswith("contract_to_spatial.") for key in state_dict):
            config["prediction_mode"] = "heatmap"
        else:
            config["prediction_mode"] = "regression"
    if "image_in_channels" not in config:
        conv_key = "image_encoder.net.0.weight"
        if conv_key in state_dict:
            config["image_in_channels"] = int(state_dict[conv_key].shape[1])
        else:
            config["image_in_channels"] = 6
    if config.get("prediction_mode") == "heatmap" and "heatmap_size" not in config:
        config["heatmap_size"] = 16
    if config.get("prediction_mode") == "heatmap" and "heatmap_xy_range_m" not in config:
        config["heatmap_xy_range_m"] = 0.04
    if config.get("prediction_mode") == "heatmap" and "heatmap_channels" not in config:
        if "heatmap_fusion.2.weight" in state_dict:
            config["heatmap_channels"] = int(state_dict["heatmap_fusion.2.weight"].shape[0])
        else:
            config["heatmap_channels"] = 3
    if config.get("prediction_mode") == "heatmap" and "heatmap_pos_weight" not in config:
        config["heatmap_pos_weight"] = 8.0
    output_dim = config.get("output_dim")
    if output_dim is None:
        head_key = "head.4.weight"
        if head_key in state_dict:
            output_dim = int(state_dict[head_key].shape[0])
        else:
            output_dim = 5
    config["output_dim"] = int(output_dim)
    model = DepthGeometryLocalizerNet.from_vocab(vocab, **config)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, vocab


def _to_rgbd_tensor(
    observation: Any,
    *,
    crop_size: int | None = None,
    resize_size: int | None = None,
    robot_state: Any = None,
) -> torch.Tensor | None:
    rgb = None
    depth = None
    for key in ("wrist_rgb", "front_rgb"):
        value = observation.get(key) if isinstance(observation, Mapping) else getattr(observation, key, None)
        if value is not None:
            rgb = np.asarray(value, dtype=np.float32)
            break
    for key in ("wrist_depth", "front_depth"):
        value = observation.get(key) if isinstance(observation, Mapping) else getattr(observation, key, None)
        if value is not None:
            depth = np.asarray(value, dtype=np.float32)
            break
    if rgb is None or depth is None:
        return None
    if depth.ndim == 3:
        depth = depth[..., 0]
    if float(depth.max()) > 1.5:
        depth = depth / 255.0
    depth = np.clip(depth, 0.0, 1.0)
    rgb = np.clip(rgb / 255.0, 0.0, 1.0)
    rgbd = np.concatenate([rgb, depth[..., None]], axis=-1).astype(np.float32)
    if crop_size is not None and crop_size > 0:
        h, w = rgbd.shape[:2]
        center_x, center_y = _roi_center_from_prior(observation, robot_state, w, h)
        crop = int(max(8, min(crop_size, min(w, h))))
        half = crop // 2
        x0 = int(round(center_x - half))
        y0 = int(round(center_y - half))
        x0 = max(0, min(x0, w - crop))
        y0 = max(0, min(y0, h - crop))
        x1 = x0 + crop
        y1 = y0 + crop
        rgbd = rgbd[y0:y1, x0:x1]
    if resize_size is not None and resize_size > 0 and (rgbd.shape[0] != resize_size or rgbd.shape[1] != resize_size):
        rgbd = _resize_rgbd(rgbd, int(resize_size))
    if rgbd.shape[-1] == 4:
        h, w = rgbd.shape[:2]
        xs = np.linspace(-1.0, 1.0, num=w, dtype=np.float32)
        ys = np.linspace(-1.0, 1.0, num=h, dtype=np.float32)
        grid_x, grid_y = np.meshgrid(xs, ys)
        rgbd = np.concatenate([rgbd, grid_x[..., None], grid_y[..., None]], axis=-1)
    return torch.from_numpy(np.transpose(rgbd, (2, 0, 1))).float()


class LearnedDepthLocalizerAdapter:
    """Runtime adapter that turns a trained checkpoint into a v2 localizer interface."""

    def __init__(self, model: DepthGeometryLocalizerNet, vocab: Mapping[str, Mapping[str, int]], *, device: str | torch.device = "cpu") -> None:
        self.model = model.to(device)
        self.model.eval()
        self.vocab = {name: dict(mapping) for name, mapping in vocab.items()}
        self.device = torch.device(device)

    @classmethod
    def from_checkpoint(cls, path: str | Path, *, device: str | torch.device = "cpu") -> "LearnedDepthLocalizerAdapter":
        model, vocab = load_depth_localizer_checkpoint(path, map_location=device)
        return cls(model, vocab, device=device)

    def _encode(self, skill_spec, stage_name: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        skill_type_id = torch.tensor([_lookup(self.vocab["skill_type"], str(getattr(skill_spec, "skill_type", "")))], dtype=torch.long, device=self.device)
        stage_id = torch.tensor([_lookup(self.vocab["stage_name"], str(stage_name))], dtype=torch.long, device=self.device)
        target_entity_id = torch.tensor([_lookup(self.vocab["entity"], str(getattr(skill_spec, "target_entity", "")))], dtype=torch.long, device=self.device)
        reference_entity_id = torch.tensor([_lookup(self.vocab["entity"], str(getattr(skill_spec, "reference_entity", "")))], dtype=torch.long, device=self.device)
        dof_vec = torch.zeros((1, len(self.vocab["controlled_dofs"])), dtype=torch.float32, device=self.device)
        for dof in getattr(skill_spec, "controlled_dofs", ()):
            idx = self.vocab["controlled_dofs"].get(str(dof))
            if idx is not None:
                dof_vec[0, int(idx)] = 1.0
        return skill_type_id, stage_id, target_entity_id, reference_entity_id, dof_vec

    def localize(self, observation, robot_state, task_spec, skill_spec, *, stage_name: str = "") -> LocalGeometryError:
        crop_size = int(getattr(skill_spec, "roi_size_px", 96) or 96)
        planner_delta = None
        if isinstance(robot_state, Mapping):
            planner_delta = robot_state.get("planner_delta_7d", None)
        if planner_delta is not None:
            try:
                delta = np.asarray(planner_delta, dtype=np.float32).reshape(-1)
                lateral_mag = float(np.linalg.norm(delta[:2]))
                crop_size = int(np.clip(crop_size + round(32.0 * min(1.0, lateral_mag / 0.01)), 8, 128))
            except Exception:
                pass
        resize_size = int(getattr(skill_spec, "roi_resize_px", 128) or 128)
        rgbd = _to_rgbd_tensor(observation, crop_size=crop_size, resize_size=resize_size, robot_state=robot_state)
        if rgbd is None:
            return LocalGeometryError(False, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, "missing_rgbd", getattr(skill_spec, "target_entity", ""), getattr(skill_spec, "reference_entity", ""), stage_name)
        skill_type_id, stage_id, target_entity_id, reference_entity_id, dof_vec = self._encode(skill_spec, stage_name)
        with torch.no_grad():
            pred = self.model.predict(
                rgbd.unsqueeze(0).to(self.device),
                skill_type_id,
                stage_id,
                target_entity_id,
                reference_entity_id,
                dof_vec,
                xy_range_m=float(getattr(skill_spec, "heatmap_xy_range_m", getattr(self.model, "heatmap_xy_range_m", 0.04))),
            )
        dx = float(pred["dx"][0].item()) if isinstance(pred["dx"], torch.Tensor) else float(pred["dx"])
        dy = float(pred["dy"][0].item()) if isinstance(pred["dy"], torch.Tensor) else float(pred["dy"])
        dz = float(pred["dz"][0].item()) if isinstance(pred["dz"], torch.Tensor) else float(pred["dz"])
        dyaw = float(pred["dyaw"][0].item()) if isinstance(pred["dyaw"], torch.Tensor) else float(pred["dyaw"])
        confidence = float(pred["confidence"][0].item()) if isinstance(pred["confidence"], torch.Tensor) else float(pred["confidence"])
        valid = confidence >= float(getattr(skill_spec, "shadow_confidence", 0.2))
        return LocalGeometryError(
            valid=valid,
            confidence=confidence,
            dx=dx,
            dy=dy,
            dz=dz,
            dyaw=dyaw,
            observability=confidence,
            fit_residual=float(max(0.0, 1.0 - confidence)),
            inlier_ratio=float(confidence),
            reason="ok" if valid else "low_confidence",
            target_entity=getattr(skill_spec, "target_entity", ""),
            reference_entity=getattr(skill_spec, "reference_entity", ""),
            stage_name=stage_name,
        )
