"""Dataset for non-privileged alignment diffusion refinement.

Expected NPZ keys are intentionally runtime-contract oriented.  Privileged
geometry may exist in the file for audit, but this dataset does not expose it to
the model by default.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class AlignmentDiffusionDataset(Dataset):
    def __init__(self, npz_path: str | Path):
        self.path = Path(npz_path)
        data = np.load(self.path, allow_pickle=False)
        self.wrist_depth = data["wrist_depth"].astype(np.float32)
        self.proprio = data["proprio"].astype(np.float32)
        self.planner_action_local = data["planner_action_local"].astype(np.float32)
        self.residual_trajectory_4d = data["residual_trajectory_4d"].astype(np.float32)
        self.force_history = data["force_history"].astype(np.float32) if "force_history" in data else None
        self.front_rgb = data["front_rgb"].astype(np.float32) if "front_rgb" in data else None
        self.wrist_rgb = data["wrist_rgb"].astype(np.float32) if "wrist_rgb" in data else None
        self.gripper_context = data["gripper_context"].astype(np.float32) if "gripper_context" in data else None
        self.progress_label = data["progress_label"].astype(np.float32) if "progress_label" in data else None
        self.risk_label = data["risk_label"].astype(np.float32) if "risk_label" in data else None
        self.stop_label = data["stop_label"].astype(np.float32) if "stop_label" in data else None
        self.bucket_id = data["bucket_id"].astype(np.int64) if "bucket_id" in data else None
        self.sample_weight = data["sample_weight"].astype(np.float32) if "sample_weight" in data else None
        self.data_aug_id = data["data_aug_id"].astype(np.int64) if "data_aug_id" in data else None

    def __len__(self) -> int:
        return int(self.proprio.shape[0])

    def _tensor_or_zeros(self, arr, idx: int, shape: tuple[int, ...]) -> torch.Tensor:
        if arr is None:
            return torch.zeros(shape, dtype=torch.float32)
        return torch.from_numpy(arr[idx]).float()

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sample = {
            "wrist_depth": torch.from_numpy(self.wrist_depth[idx]).float(),
            "force_history": self._tensor_or_zeros(self.force_history, idx, (32, 6)),
            "proprio": torch.from_numpy(self.proprio[idx]).float(),
            "planner_action_local": torch.from_numpy(self.planner_action_local[idx]).float(),
            "residual_trajectory_4d": torch.from_numpy(self.residual_trajectory_4d[idx]).float(),
            "front_rgb": self._tensor_or_zeros(self.front_rgb, idx, (3, 96, 96)),
            "wrist_rgb": self._tensor_or_zeros(self.wrist_rgb, idx, (3, 96, 96)),
            "gripper_context": self._tensor_or_zeros(self.gripper_context, idx, (4,)),
            "progress_label": self._tensor_or_zeros(self.progress_label, idx, (3,)),
            "risk_label": self._tensor_or_zeros(self.risk_label, idx, (1,)).reshape(1),
            "stop_label": self._tensor_or_zeros(self.stop_label, idx, (1,)).reshape(1),
        }
        if self.bucket_id is not None:
            sample["bucket_id"] = torch.tensor(int(self.bucket_id[idx]), dtype=torch.long)
        if self.sample_weight is not None:
            sample["sample_weight"] = torch.tensor(float(self.sample_weight[idx]), dtype=torch.float32)
        if self.data_aug_id is not None:
            sample["data_aug_id"] = torch.tensor(int(self.data_aug_id[idx]), dtype=torch.long)
        return sample
