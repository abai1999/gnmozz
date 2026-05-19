"""Near/micro-only dataset for alignment_v3_direct_local_controller."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset


class AlignmentV3DirectLocalDataset(Dataset):
    """Dataset for direct local near-contact controller training/shadow audit."""

    def __init__(self, npz_path: str, stage_bucket_filter: Optional[List[str]] = None):
        data = np.load(Path(npz_path), allow_pickle=True)
        self._data = {k: np.asarray(data[k]) for k in data.files}
        self.length = int(self._data["current_to_target_delta_local"].shape[0])

        if stage_bucket_filter is not None:
            buckets = self._data.get("stage_bucket", np.array(["unknown"] * self.length))
            mask = np.array([b in stage_bucket_filter for b in buckets], dtype=bool)
        else:
            mask = np.ones(self.length, dtype=bool)
        self._indices = np.where(mask)[0]

    def __len__(self) -> int:
        return int(self._indices.size)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor | str]:
        i = int(self._indices[idx])

        def _arr(key: str, dtype=np.float32):
            return torch.from_numpy(np.asarray(self._data[key][i], dtype=dtype))

        wrist_depth = _arr("wrist_depth")
        if wrist_depth.ndim == 2:
            wrist_depth = wrist_depth.unsqueeze(0)

        if "planner_base_action_local" in self._data:
            planner_action = _arr("planner_base_action_local")[:6]
        elif "planner_action_local" in self._data:
            planner_action = _arr("planner_action_local")[:6]
        elif "planner_base_action_local_raw" in self._data:
            planner_action = _arr("planner_base_action_local_raw")[:6]
        elif "base_action" in self._data:
            planner_action = _arr("base_action")[:6]
        else:
            planner_action = torch.zeros((6,), dtype=torch.float32)

        out: Dict[str, torch.Tensor | str] = {
            "row_index": torch.tensor(i, dtype=torch.long),
            "wrist_depth": wrist_depth,
            "force_history": _arr("force_history"),
            "proprio": _arr("proprio"),
            "planner_action_local": planner_action,
            "current_to_target_delta_local": _arr("current_to_target_delta_local")[:6],
            "runtime_current_to_target_delta_local": _arr("runtime_current_to_target_delta_local")[:6]
            if "runtime_current_to_target_delta_local" in self._data
            else _arr("current_to_target_delta_local")[:6],
            "teacher_current_to_target_delta_local": _arr("teacher_current_to_target_delta_local")[:6]
            if "teacher_current_to_target_delta_local" in self._data
            else _arr("current_to_target_delta_local")[:6],
            "raw_learned_predictor_delta_local": _arr("raw_learned_predictor_delta_local")[:6]
            if "raw_learned_predictor_delta_local" in self._data
            else _arr("current_to_target_delta_local")[:6],
            "target_residual_local_4d": _arr("target_residual_local_4d"),
            "target_residual_local_6d": _arr("target_residual_local_6d"),
            "target_post_xy_error": _arr("target_post_xy_error").reshape(()),
            "target_post_z_error": _arr("target_post_z_error").reshape(()),
            "target_post_yaw_error": _arr("target_post_yaw_error").reshape(()),
            "current_xy_error": _arr("current_xy_error").reshape(()),
            "current_z_error": _arr("current_z_error").reshape(()),
            "current_yaw_error": _arr("current_yaw_error").reshape(()),
            "target_improves_xy": _arr("target_improves_xy").reshape(()),
            "target_improves_z": _arr("target_improves_z").reshape(()),
            "target_improves_yaw": _arr("target_improves_yaw").reshape(()),
            "invalid_risk_proxy": _arr("invalid_risk_proxy").reshape(()),
            "overshoot_proxy": _arr("overshoot_proxy").reshape(()),
            "stage_bucket": str(self._data["stage_bucket"][i]),
        }
        for key in (
            "gripper_context",
            "has_object_in_hand",
            "substage_id",
            "contact_state",
            "stage_target_mode",
            "depth_proximity",
            "planner_close_intent",
            "planner_close_intent_strength",
            "runtime_target_delta_source",
            "runtime_target_delta_context_mode",
        ):
            if key in self._data:
                val = self._data[key][i]
                if key in ("gripper_context",):
                    out[key] = torch.from_numpy(np.asarray(val, dtype=np.float32))
                elif key in ("runtime_target_delta_source", "runtime_target_delta_context_mode"):
                    out[key] = str(val)
                elif key in ("substage_id", "contact_state", "stage_target_mode"):
                    out[key] = torch.tensor(int(val), dtype=torch.long)
                else:
                    out[key] = torch.tensor(float(val), dtype=torch.float32)
        if "episode_index" in self._data:
            out["episode_index"] = torch.tensor(int(self._data["episode_index"][i]), dtype=torch.long)
        if "step_index" in self._data:
            out["step_index"] = torch.tensor(int(self._data["step_index"][i]), dtype=torch.long)
        return out
