"""Dataset for verified-only phase1/phase2 alignment student vNext."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class AlignmentTCStudentVNextDataset(Dataset):
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
        self.teacher_contact_repr = data["teacher_contact_repr"].astype(np.float32)
        self.teacher_residual_action_4d = data["teacher_residual_action_4d"].astype(np.float32)
        self.teacher_residual_trajectory_4d = data["teacher_residual_trajectory_4d"].astype(np.float32)
        self.progress_label = data["teacher_progress_label"].astype(np.float32)
        self.risk_label = data["teacher_risk_label"].astype(np.float32)
        self.stop_label = data["teacher_stop_label"].astype(np.float32)
        self.confidence_label = data["teacher_confidence_label"].astype(np.float32)
        self.sample_weight = data["sample_weight"].astype(np.float32)
        self.phase_id = data["phase_id"].astype(np.int64)
        self.stage_bucket_id = data["stage_bucket_id"].astype(np.int64)
        self.yaw_direction_label = data["yaw_direction_label"].astype(np.int64)
        self.yaw_imitation_enabled = data["yaw_imitation_enabled"].astype(np.float32)
        self.verified_positive = data["verified_positive"].astype(np.float32)
        zeros = np.zeros_like(self.sample_weight, dtype=np.float32)
        self.teacher_close_ready = data["teacher_close_ready"].astype(np.float32) if "teacher_close_ready" in data else zeros.copy()
        self.teacher_close_ready_all = data["teacher_close_ready_all"].astype(np.float32) if "teacher_close_ready_all" in data else zeros.copy()
        self.teacher_truth_handoff_ready = (
            data["teacher_truth_handoff_ready"].astype(np.float32) if "teacher_truth_handoff_ready" in data else zeros.copy()
        )
        self.teacher_close_ready_score = (
            data["teacher_close_ready_score"].astype(np.float32) if "teacher_close_ready_score" in data else zeros.copy()
        )
        self.close_ready_bridge_mask = (
            data["close_ready_bridge_mask"].astype(np.float32) if "close_ready_bridge_mask" in data else zeros.copy()
        )
        self.close_ready_exact_mask = (
            data["close_ready_exact_mask"].astype(np.float32) if "close_ready_exact_mask" in data else zeros.copy()
        )

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
        if ten.shape[-2:] != (96, 96):
            ten = F.interpolate(ten.unsqueeze(0), size=(96, 96), mode="bilinear", align_corners=False).squeeze(0)
        if ten.max() > 1.5:
            ten = ten / 255.0
        return ten.contiguous().float()

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "wrist_depth": torch.from_numpy(self.wrist_depth[idx]).float(),
            "force_history": self._tensor_or_zeros(self.force_history, idx, (32, 6)),
            "proprio": torch.from_numpy(self.proprio[idx]).float(),
            "planner_action_local": torch.from_numpy(self.planner_action_local[idx]).float(),
            "front_rgb": self._rgb_tensor(self.front_rgb, idx),
            "wrist_rgb": self._rgb_tensor(self.wrist_rgb, idx),
            "gripper_context": self._tensor_or_zeros(self.gripper_context, idx, (4,)),
            "teacher_target_delta_local_6d": torch.from_numpy(self.teacher_target_delta_local_6d[idx]).float(),
            "teacher_contact_repr": torch.from_numpy(self.teacher_contact_repr[idx]).float(),
            "teacher_residual_action_4d": torch.from_numpy(self.teacher_residual_action_4d[idx]).float(),
            "teacher_residual_trajectory_4d": torch.from_numpy(self.teacher_residual_trajectory_4d[idx]).float(),
            "teacher_progress_label": torch.from_numpy(self.progress_label[idx]).float(),
            "teacher_risk_label": torch.tensor(float(self.risk_label[idx]), dtype=torch.float32),
            "teacher_stop_label": torch.tensor(float(self.stop_label[idx]), dtype=torch.float32),
            "teacher_confidence_label": torch.tensor(float(self.confidence_label[idx]), dtype=torch.float32),
            "sample_weight": torch.tensor(float(self.sample_weight[idx]), dtype=torch.float32),
            "phase_id": torch.tensor(int(self.phase_id[idx]), dtype=torch.long),
            "stage_bucket_id": torch.tensor(int(self.stage_bucket_id[idx]), dtype=torch.long),
            "yaw_direction_label": torch.tensor(int(self.yaw_direction_label[idx]), dtype=torch.long),
            "yaw_imitation_enabled": torch.tensor(float(self.yaw_imitation_enabled[idx]), dtype=torch.float32),
            "verified_positive": torch.tensor(float(self.verified_positive[idx]), dtype=torch.float32),
            "teacher_close_ready": torch.tensor(float(self.teacher_close_ready[idx]), dtype=torch.float32),
            "teacher_close_ready_all": torch.tensor(float(self.teacher_close_ready_all[idx]), dtype=torch.float32),
            "teacher_truth_handoff_ready": torch.tensor(float(self.teacher_truth_handoff_ready[idx]), dtype=torch.float32),
            "teacher_close_ready_score": torch.tensor(float(self.teacher_close_ready_score[idx]), dtype=torch.float32),
            "close_ready_bridge_mask": torch.tensor(float(self.close_ready_bridge_mask[idx]), dtype=torch.float32),
            "close_ready_exact_mask": torch.tensor(float(self.close_ready_exact_mask[idx]), dtype=torch.float32),
        }
