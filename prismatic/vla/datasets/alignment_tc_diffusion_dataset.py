"""Dataset for target-conditioned diffusion alignment.

Runtime inputs are non-privileged.  Privileged geometry appears only as labels
for target estimation, teacher trajectory imitation, ranking, and audit.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class AlignmentTCDiffusionDataset(Dataset):
    def __init__(self, npz_path: str | Path):
        self.path = Path(npz_path)
        data = np.load(self.path, allow_pickle=False)
        self.wrist_depth = data["wrist_depth"].astype(np.float32)
        self.proprio = data["proprio"].astype(np.float32)
        self.planner_action_local = data["planner_action_local"].astype(np.float32)
        if self.planner_action_local.ndim == 2 and self.planner_action_local.shape[1] > 6:
            self.planner_action_local = self.planner_action_local[:, :6]
        self.force_history = data["force_history"].astype(np.float32) if "force_history" in data else None
        self.front_rgb = data["front_rgb"].astype(np.float32) if "front_rgb" in data else None
        self.wrist_rgb = data["wrist_rgb"].astype(np.float32) if "wrist_rgb" in data else None
        self.gripper_context = data["gripper_context"].astype(np.float32) if "gripper_context" in data else None

        self.teacher_target_delta_local_6d = data["teacher_target_delta_local_6d"].astype(np.float32)
        self.contact_heatmap_label = data["contact_heatmap_label"].astype(np.float32)
        self.target_confidence_label = data["target_confidence_label"].astype(np.float32)
        self.best_residual_trajectory_4d = data["best_residual_trajectory_4d"].astype(np.float32)
        self.teacher_residual_trajectory_4d = (
            data["teacher_residual_trajectory_4d"].astype(np.float32)
            if "teacher_residual_trajectory_4d" in data
            else None
        )
        self.progress_label = data["progress_label"].astype(np.float32) if "progress_label" in data else None
        self.risk_label = data["risk_label"].astype(np.float32) if "risk_label" in data else None
        self.stop_label = data["stop_label"].astype(np.float32) if "stop_label" in data else None
        self.sample_weight = data["sample_weight"].astype(np.float32) if "sample_weight" in data else None
        self.best_candidate_index = data["best_candidate_index"].astype(np.int64) if "best_candidate_index" in data else None
        self.candidate_score = data["candidate_score"].astype(np.float32) if "candidate_score" in data else None
        self.stage_bucket = data["stage_bucket"].astype(str) if "stage_bucket" in data else None
        self.alignment_phase = data["alignment_phase"].astype(str) if "alignment_phase" in data else None
        self.target_phase = data["target_phase"].astype(str) if "target_phase" in data else None

    def __len__(self) -> int:
        return int(self.proprio.shape[0])

    def _tensor_or_zeros(self, arr, idx: int, shape: tuple[int, ...]) -> torch.Tensor:
        if arr is None:
            return torch.zeros(shape, dtype=torch.float32)
        return torch.from_numpy(arr[idx]).float()

    @staticmethod
    def _rgb_tensor(arr, idx: int) -> torch.Tensor:
        if arr is None:
            return torch.zeros((3, 96, 96), dtype=torch.float32)
        ten = torch.from_numpy(arr[idx]).float()
        if ten.ndim == 3 and ten.shape[-1] == 3:
            ten = ten.permute(2, 0, 1)
        elif ten.ndim == 3 and ten.shape[0] == 3:
            pass
        else:
            ten = ten.reshape(3, ten.shape[-2], ten.shape[-1]) if ten.numel() == 3 * ten.shape[-2] * ten.shape[-1] else ten
        if ten.ndim != 3 or ten.shape[0] != 3:
            raise ValueError(f"expected rgb tensor with 3 channels, got shape {tuple(ten.shape)}")
        if ten.shape[-1] != 96 or ten.shape[-2] != 96:
            ten = F.interpolate(ten.unsqueeze(0), size=(96, 96), mode="bilinear", align_corners=False).squeeze(0)
        if ten.max() > 1.5:
            ten = ten / 255.0
        return ten.contiguous().float()

    @staticmethod
    def _scalar(value, default: float = 0.0) -> float:
        arr = np.asarray(value, dtype=np.float32).reshape(-1)
        if arr.size == 0:
            return float(default)
        return float(arr[0])

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sample = {
            "wrist_depth": torch.from_numpy(self.wrist_depth[idx]).float(),
            "force_history": self._tensor_or_zeros(self.force_history, idx, (32, 6)),
            "proprio": torch.from_numpy(self.proprio[idx]).float(),
            "planner_action_local": torch.from_numpy(self.planner_action_local[idx]).float(),
            "front_rgb": self._rgb_tensor(self.front_rgb, idx),
            "wrist_rgb": self._rgb_tensor(self.wrist_rgb, idx),
            "gripper_context": self._tensor_or_zeros(self.gripper_context, idx, (4,)),
            "teacher_target_delta_local_6d": torch.from_numpy(self.teacher_target_delta_local_6d[idx]).float(),
            "contact_heatmap_label": torch.from_numpy(self.contact_heatmap_label[idx]).float(),
            "target_confidence_label": torch.tensor(self._scalar(self.target_confidence_label[idx]), dtype=torch.float32),
            "best_residual_trajectory_4d": torch.from_numpy(self.best_residual_trajectory_4d[idx]).float(),
            "progress_label": self._tensor_or_zeros(self.progress_label, idx, (3,)),
            "risk_label": self._tensor_or_zeros(self.risk_label, idx, (1,)).reshape(1),
            "stop_label": self._tensor_or_zeros(self.stop_label, idx, (1,)).reshape(1),
            "sample_weight": torch.tensor(
                self._scalar(1.0 if self.sample_weight is None else self.sample_weight[idx], default=1.0),
                dtype=torch.float32,
            ),
        }
        if self.best_candidate_index is not None:
            sample["best_candidate_index"] = torch.tensor(int(self.best_candidate_index[idx]), dtype=torch.long)
        if self.candidate_score is not None:
            scores = np.asarray(self.candidate_score[idx], dtype=np.float32).reshape(-1)
            sample["candidate_score"] = torch.tensor(
                float(np.min(scores)) if scores.size else 0.0,
                dtype=torch.float32,
            )
        if self.stage_bucket is not None:
            sample["stage_bucket_id"] = torch.tensor(
                {"near_contact_refine": 0, "micro_contact_refine": 1, "broad_near": 2}.get(
                    str(self.stage_bucket[idx]), -1
                ),
                dtype=torch.long,
            )
        if self.alignment_phase is not None:
            sample["alignment_phase_id"] = torch.tensor(
                {"grasp_commit": 0, "insert_commit": 1, "commit_target": 2}.get(
                    str(self.alignment_phase[idx]), -1
                ),
                dtype=torch.long,
            )
        if self.target_phase is not None:
            sample["target_phase_id"] = torch.tensor(
                {"grasp_commit": 0, "insert_commit": 1, "commit_target": 2}.get(
                    str(self.target_phase[idx]), -1
                ),
                dtype=torch.long,
            )
        if self.teacher_residual_trajectory_4d is not None:
            sample["teacher_residual_trajectory_4d"] = torch.from_numpy(self.teacher_residual_trajectory_4d[idx]).float()
        return sample
