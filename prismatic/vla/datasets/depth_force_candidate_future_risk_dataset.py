"""
depth_force_candidate_future_risk_dataset.py

Map-style dataset for candidate-conditioned future-risk supervision.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import torch
from torch.utils.data import Dataset


class DepthForceCandidateFutureRiskDataset(Dataset):
    def __init__(self, npz_path: str):
        raw = np.load(Path(npz_path), allow_pickle=True)
        self.data = {k: np.asarray(raw[k]) for k in raw.files}
        self.length = int(self.data["candidate_actions_local"].shape[0])

    def __len__(self) -> int:
        return self.length

    def _pick_candidate_array(self, *keys: str, default: np.ndarray | None = None) -> np.ndarray:
        for key in keys:
            if key in self.data:
                return np.asarray(self.data[key])
        if default is not None:
            return np.asarray(default)
        raise KeyError(f"none of the candidate keys exist: {keys}")

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        front = torch.from_numpy(self.data["front_rgb"][idx].astype(np.float32).transpose(2, 0, 1)) / 255.0
        wrist = torch.from_numpy(self.data["wrist_rgb"][idx].astype(np.float32).transpose(2, 0, 1)) / 255.0
        depth = torch.from_numpy(self.data["wrist_depth"][idx].astype(np.float32))
        if depth.ndim == 2:
            depth = depth.unsqueeze(0)
        force_arr = self.data.get("force_history_normalized", self.data.get("force_history", self.data.get("ft_hist")))
        if force_arr is None:
            force_arr = np.zeros((self.length, 32, 6), dtype=np.float32)
        force_hist = torch.from_numpy(np.asarray(force_arr[idx], dtype=np.float32))
        if force_hist.ndim == 2 and force_hist.shape[-1] == 6:
            pass
        elif force_hist.ndim == 2 and force_hist.shape[0] == 6:
            force_hist = force_hist.transpose(0, 1)
        proprio = torch.from_numpy(self.data["proprio"][idx].astype(np.float32))
        planner = torch.from_numpy(
            np.asarray(
                self.data.get("planner_base_action_local_raw", self.data.get("planner_base_action_local"))[idx],
                dtype=np.float32,
            )
        )
        candidates = torch.from_numpy(self.data["candidate_actions_local"][idx].astype(np.float32))
        mask_default = np.ones((self.data["candidate_actions_local"].shape[1],), dtype=np.float32)
        mask = torch.from_numpy(self.data.get("candidate_mask", mask_default)[idx].astype(np.float32))
        stage_token = torch.tensor(int(self.data.get("stage_token", self.data.get("substage_id", np.zeros((self.length,), dtype=np.int64)))[idx]), dtype=torch.long)
        contact_phase = torch.tensor(int(self.data.get("contact_state", np.zeros((self.length,), dtype=np.int64))[idx]), dtype=torch.long)
        depth_prox = torch.tensor(float(self.data.get("depth_proximity", np.zeros((self.length,), dtype=np.float32))[idx]), dtype=torch.float32)
        gripper_state = torch.tensor(float(self.data.get("gripper_state", np.zeros((self.length,), dtype=np.float32))[idx]), dtype=torch.float32)
        candidate_baseline_index = torch.tensor(int(self.data["candidate_future_risk_baseline_index"][idx]), dtype=torch.long)
        candidate_geom_index = torch.tensor(int(self.data["candidate_future_risk_geom_index"][idx]), dtype=torch.long)
        candidate_best_index = torch.tensor(int(self.data["candidate_future_risk_best_index"][idx]), dtype=torch.long)
        geometry_key = "candidate_privileged_geometry_cost" if "candidate_privileged_geometry_cost" in self.data else "candidate_geometry_cost"
        geometry_norm_key = (
            "candidate_privileged_geometry_cost_norm"
            if "candidate_privileged_geometry_cost_norm" in self.data
            else "candidate_geometry_cost_norm"
        )
        risk_key = "candidate_risk_cost"
        risk_norm_key = "candidate_risk_cost_norm"
        if risk_key not in self.data:
            if "candidate_risk_score" in self.data:
                risk_key = "candidate_risk_score"
            elif "candidate_future_risk_score" in self.data:
                risk_key = "candidate_future_risk_score"
            elif "candidate_future_risk_delta" in self.data:
                risk_key = "candidate_future_risk_delta"
        if risk_norm_key not in self.data:
            if "candidate_risk_score_norm" in self.data:
                risk_norm_key = "candidate_risk_score_norm"
            elif "candidate_future_risk_score_norm" in self.data:
                risk_norm_key = "candidate_future_risk_score_norm"
            elif "candidate_future_risk_delta_norm" in self.data:
                risk_norm_key = "candidate_future_risk_delta_norm"
        risk_arr = self._pick_candidate_array(
            risk_key,
            "candidate_future_risk_score",
            "candidate_future_risk_delta",
            default=np.zeros_like(self.data[geometry_key], dtype=np.float32),
        )
        risk_norm_arr = self._pick_candidate_array(
            risk_norm_key,
            "candidate_future_risk_score_norm",
            "candidate_future_risk_delta_norm",
            default=risk_arr.astype(np.float32),
        )
        total_cost_arr = self._pick_candidate_array(
            "candidate_total_cost",
            "candidate_future_risk_score",
            "candidate_future_risk_delta",
            default=(self.data[geometry_key].astype(np.float32) + risk_arr.astype(np.float32)),
        )
        total_cost_norm_arr = self._pick_candidate_array(
            "candidate_total_cost_norm",
            default=total_cost_arr.astype(np.float32),
        )
        return {
            "front_rgb": front,
            "wrist_rgb": wrist,
            "wrist_depth": depth,
            "force_history": force_hist,
            "proprio": proprio,
            "planner_base_action_local": planner,
            "candidate_actions_local": candidates,
            "candidate_mask": mask,
            "stage_token": stage_token,
            "contact_phase": contact_phase,
            "depth_proximity": depth_prox,
            "gripper_state": gripper_state,
            "episode_index": torch.tensor(int(self.data["episode_index"][idx]), dtype=torch.long),
            "yaw_opportunity_label": torch.tensor(float(self.data.get("yaw_opportunity_label", np.zeros((self.length,), dtype=np.float32))[idx]), dtype=torch.float32),
            "yaw_augmentation_applied": torch.tensor(float(self.data.get("yaw_augmentation_applied", np.zeros((self.length,), dtype=np.float32))[idx]), dtype=torch.float32),
            "candidate_baseline_index": candidate_baseline_index,
            "candidate_geom_index": candidate_geom_index,
            "candidate_best_index": candidate_best_index,
            "candidate_geometry_cost": torch.from_numpy(self.data[geometry_key][idx].astype(np.float32)),
            "candidate_geometry_cost_norm": torch.from_numpy(
                self.data.get(geometry_norm_key, self.data[geometry_key])[idx].astype(np.float32)
            ),
            "candidate_risk_cost": torch.from_numpy(risk_arr[idx].astype(np.float32)),
            "candidate_risk_cost_norm": torch.from_numpy(risk_norm_arr[idx].astype(np.float32)),
            "candidate_total_cost": torch.from_numpy(total_cost_arr[idx].astype(np.float32)),
            "candidate_total_cost_norm": torch.from_numpy(total_cost_norm_arr[idx].astype(np.float32)),
            "candidate_future_risk_score": torch.from_numpy(self.data["candidate_future_risk_score"][idx].astype(np.float32)),
            "candidate_future_risk_delta": torch.from_numpy(self.data["candidate_future_risk_delta"][idx].astype(np.float32)),
            "candidate_future_risk_label": torch.from_numpy(self.data["candidate_future_risk_label"][idx].astype(np.float32)),
            "candidate_future_risk_nonincrease_label": torch.from_numpy(self.data["candidate_future_risk_nonincrease_label"][idx].astype(np.float32)),
            "candidate_future_contact_risk": torch.from_numpy(self.data["candidate_future_contact_risk"][idx].astype(np.float32)),
            "candidate_future_force_spike_risk": torch.from_numpy(self.data["candidate_future_force_spike_risk"][idx].astype(np.float32)),
            "candidate_future_jam_risk": torch.from_numpy(self.data["candidate_future_jam_risk"][idx].astype(np.float32)),
            "candidate_future_motion_stall_risk": torch.from_numpy(self.data["candidate_future_motion_stall_risk"][idx].astype(np.float32)),
            "candidate_future_kinematic_invalid_risk": torch.from_numpy(self.data["candidate_future_kinematic_invalid_risk"][idx].astype(np.float32)),
            "candidate_future_action_range_invalid_risk": torch.from_numpy(self.data["candidate_future_action_range_invalid_risk"][idx].astype(np.float32)),
            "candidate_future_risk_mask": torch.from_numpy(self.data["candidate_future_risk_mask"][idx].astype(np.float32)),
        }
