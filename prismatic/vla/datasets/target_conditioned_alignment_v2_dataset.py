"""Dataset for Target-Conditioned Alignment v2.

Loads the v2 NPZ built by build_target_conditioned_alignment_v2_dataset.py.
Supports per-bucket filtering for near/micro-first training.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset


class TargetConditionedAlignmentV2Dataset(Dataset):
    def __init__(
        self,
        npz_path: str,
        stage_bucket_filter: Optional[List[str]] = None,
    ):
        data = np.load(Path(npz_path), allow_pickle=True)
        self._data = {k: np.asarray(data[k]) for k in data.files}
        self.length = int(self._data["current_to_target_delta_local"].shape[0])
        self.K = int(self._data["proposal_actions_local"].shape[1])

        # Build row mask
        if stage_bucket_filter is not None:
            buckets = self._data.get("stage_bucket", np.array(["unknown"] * self.length))
            self._mask = np.array([b in stage_bucket_filter for b in buckets], dtype=bool)
        else:
            self._mask = np.ones(self.length, dtype=bool)
        self._indices = np.where(self._mask)[0]

    def __len__(self) -> int:
        return int(self._indices.size)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        i = int(self._indices[idx])

        def _arr(key, dtype=np.float32):
            return torch.from_numpy(np.asarray(self._data[key][i], dtype=dtype))

        wrist_depth = _arr("wrist_depth")
        if wrist_depth.ndim == 2:
            wrist_depth = wrist_depth.unsqueeze(0)

        return {
            "row_index": torch.tensor(i, dtype=torch.long),
            "wrist_depth": wrist_depth,
            "force_history": _arr("force_history"),
            "proprio": _arr("proprio"),
            "planner_base_action_local": _arr("planner_base_action_local")[:6],
            "current_to_target_delta_local": _arr("current_to_target_delta_local")[:6],
            "proposal_actions": _arr("proposal_actions_local"),      # (K, 6)
            "post_candidate_delta": _arr("post_candidate_delta_local"),  # (K, 6)
            "xy_improvement": _arr("xy_improvement").squeeze(),       # (K,)
            "z_improvement": _arr("z_improvement").squeeze(),
            "yaw_improvement": _arr("yaw_improvement").squeeze(),
            "geometry_improvement": _arr("geometry_improvement").squeeze(),
            "current_xy_error": _arr("current_xy_error").squeeze(),
            "current_z_error": _arr("current_z_error").squeeze(),
            "current_yaw_error": _arr("current_yaw_error").squeeze(),
            "post_xy_error": _arr("post_xy_error").squeeze(),
            "post_z_error": _arr("post_z_error").squeeze(),
            "post_yaw_error": _arr("post_yaw_error").squeeze(),
            "overshoot_xy": _arr("overshoot_xy").squeeze(),
            "overshoot_z": _arr("overshoot_z").squeeze(),
            "overshoot_yaw": _arr("overshoot_yaw").squeeze(),
            "stage_bucket": str(self._data["stage_bucket"][i]),
            "best_stage_action_index": torch.tensor(int(self._data["best_stage_action_index"][i]), dtype=torch.long),
            "best_near_action_index": torch.tensor(int(self._data["best_near_action_index"][i]), dtype=torch.long),
            "best_micro_action_index": torch.tensor(int(self._data["best_micro_action_index"][i]), dtype=torch.long),
            "candidate_valid_mask": _arr("candidate_valid_mask").squeeze(),
            # Optional: for final_full comparison
            "front_rgb": _arr("front_rgb") if "front_rgb" in self._data else torch.zeros(3, 128, 128),
            "wrist_rgb": _arr("wrist_rgb") if "wrist_rgb" in self._data else torch.zeros(3, 128, 128),
        }
