"""
depth_force_local_proposal_dataset.py

Map-style dataset for state-conditioned local proposal supervision.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import torch
from torch.utils.data import Dataset


MODE_TO_ID = {
    "planner": 0,
    "align_refine": 1,
    "near_hold": 2,
    "contact_backoff": 3,
    "kinematic_hold": 4,
}


class DepthForceLocalProposalDataset(Dataset):
    def __init__(self, npz_path: str, proposal_cache_npz: str | None = None):
        raw = np.load(Path(npz_path), allow_pickle=True)
        self.data = {k: np.asarray(raw[k]) for k in raw.files}
        if proposal_cache_npz:
            cache = np.load(Path(proposal_cache_npz), allow_pickle=True)
            cache_data = {k: np.asarray(cache[k]) for k in cache.files}
            for key in (
                "proposal_actions_local",
                "proposal_scores_init",
                "proposal_geometry_cost",
                "proposal_risk_cost",
                "proposal_geometry_gain",
                "proposal_risk_delta",
                "proposal_pareto_mask",
                "proposal_budget_mask",
                "proposal_baseline_index",
                "proposal_geom_top1_index",
                "proposal_best_safe_index",
                "proposal_best_soft_index",
                "proposal_best_budget_index",
                "proposal_target_mode",
                "proposal_target_source",
                "proposal_target_delta_local",
                "proposal_safe_target_action_local",
                "row_index",
                "episode_index",
                "step_index",
            ):
                if key in cache_data:
                    self.data[key] = np.asarray(cache_data[key])
        self.length = int(self.data["proposal_actions_local"].shape[0])

    def __len__(self) -> int:
        return self.length

    def _pick(self, *keys: str, default: np.ndarray | None = None) -> np.ndarray:
        for key in keys:
            if key in self.data:
                return np.asarray(self.data[key])
        if default is not None:
            return np.asarray(default)
        raise KeyError(f"none of the keys exist: {keys}")

    def _image(self, key: str, idx: int, shape: tuple[int, ...], channel_first: bool = False) -> torch.Tensor:
        arr = self.data.get(key)
        if arr is None:
            arr = np.zeros(shape, dtype=np.float32)
        x = np.asarray(arr[idx], dtype=np.float32)
        if x.ndim == 3 and not channel_first:
            x = x.transpose(2, 0, 1)
        return torch.from_numpy(x.astype(np.float32))

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
        proprio = torch.from_numpy(self.data.get("proprio", np.zeros((self.length, 15), dtype=np.float32))[idx].astype(np.float32))
        planner = torch.from_numpy(
            np.asarray(
                self.data.get("planner_base_action_local_raw", self.data.get("planner_base_action_local"))[idx],
                dtype=np.float32,
            )
        )[:6]
        candidate_mask = self.data.get("proposal_pareto_mask", None)
        if candidate_mask is None:
            candidate_mask = np.ones((self.data["proposal_actions_local"].shape[1],), dtype=np.float32)
        candidate_mask = torch.from_numpy(np.asarray(candidate_mask[idx], dtype=np.float32))
        stage_token = torch.tensor(int(self.data.get("stage_token", self.data.get("substage_id", np.zeros((self.length,), dtype=np.int64)))[idx]), dtype=torch.long)
        contact_phase = torch.tensor(int(self.data.get("contact_state", np.zeros((self.length,), dtype=np.int64))[idx]), dtype=torch.long)
        depth_prox = torch.tensor(float(self.data.get("depth_proximity", np.zeros((self.length,), dtype=np.float32))[idx]), dtype=torch.float32)
        gripper_state = torch.tensor(float(self.data.get("gripper_state", np.zeros((self.length,), dtype=np.float32))[idx]), dtype=torch.float32)

        proposal_actions = torch.from_numpy(self.data["proposal_actions_local"][idx].astype(np.float32))
        proposal_geometry_cost = torch.from_numpy(self.data["proposal_geometry_cost"][idx].astype(np.float32))
        proposal_risk_cost = torch.from_numpy(self.data["proposal_risk_cost"][idx].astype(np.float32))
        proposal_geometry_gain = torch.from_numpy(self.data["proposal_geometry_gain"][idx].astype(np.float32))
        proposal_risk_delta = torch.from_numpy(self.data["proposal_risk_delta"][idx].astype(np.float32))
        proposal_pareto_mask = torch.from_numpy(self.data["proposal_pareto_mask"][idx].astype(np.float32))
        proposal_budget_mask = torch.from_numpy(self.data["proposal_budget_mask"][idx].astype(np.float32))
        proposal_baseline_index = torch.tensor(int(self.data["proposal_baseline_index"][idx]), dtype=torch.long)
        proposal_geom_top1_index = torch.tensor(int(self.data["proposal_geom_top1_index"][idx]), dtype=torch.long)
        proposal_best_safe_index = torch.tensor(int(self.data["proposal_best_safe_index"][idx]), dtype=torch.long)
        proposal_best_soft_index = torch.tensor(int(self.data["proposal_best_soft_index"][idx]), dtype=torch.long)
        proposal_best_budget_index = torch.tensor(int(self.data["proposal_best_budget_index"][idx]), dtype=torch.long)

        target_mode = str(self.data.get("proposal_target_mode", np.asarray(["planner"] * self.length, dtype="U32"))[idx])
        target_source = str(self.data.get("proposal_target_source", np.asarray(["unknown"] * self.length, dtype="U64"))[idx])
        target_mode_id = MODE_TO_ID.get(target_mode, -1)

        current_pose = self.data.get("privileged_current_pose_7d", self.data.get("current_pose_7d", np.zeros((self.length, 7), dtype=np.float32)))
        target_pose = self.data.get(
            "privileged_motion_target_pose_7d",
            self.data.get("privileged_basin_center_pose_7d", self.data.get("motion_target_pose_7d", np.zeros((self.length, 7), dtype=np.float32))),
        )
        target_delta = self.data.get("proposal_target_delta_local", np.zeros((self.length, 6), dtype=np.float32))
        safe_target = self.data.get("proposal_safe_target_action_local", np.zeros((self.length, 6), dtype=np.float32))
        yaw_aug = self.data.get("yaw_augmentation_applied", np.zeros((self.length,), dtype=np.float32))
        yaw_opp = self.data.get("yaw_opportunity_label", np.zeros((self.length,), dtype=np.float32))

        return {
            "row_index": torch.tensor(int(idx), dtype=torch.long),
            "front_rgb": front,
            "wrist_rgb": wrist,
            "wrist_depth": depth,
            "force_history": force_hist,
            "proprio": proprio,
            "planner_base_action_local": planner,
            "candidate_mask": candidate_mask,
            "stage_token": stage_token,
            "contact_phase": contact_phase,
            "depth_proximity": depth_prox,
            "gripper_state": gripper_state,
            "episode_index": torch.tensor(int(self.data["episode_index"][idx]), dtype=torch.long),
            "step_index": torch.tensor(int(self.data["step_index"][idx]), dtype=torch.long),
            "proposal_actions_local": proposal_actions,
            "proposal_geometry_cost": proposal_geometry_cost,
            "proposal_risk_cost": proposal_risk_cost,
            "proposal_geometry_gain": proposal_geometry_gain,
            "proposal_risk_delta": proposal_risk_delta,
            "proposal_pareto_mask": proposal_pareto_mask,
            "proposal_budget_mask": proposal_budget_mask,
            "proposal_baseline_index": proposal_baseline_index,
            "proposal_geom_top1_index": proposal_geom_top1_index,
            "proposal_best_safe_index": proposal_best_safe_index,
            "proposal_best_soft_index": proposal_best_soft_index,
            "proposal_best_budget_index": proposal_best_budget_index,
            "proposal_target_mode": target_mode,
            "proposal_target_mode_id": torch.tensor(target_mode_id, dtype=torch.long),
            "proposal_target_source": target_source,
            "proposal_target_delta_local": torch.from_numpy(np.asarray(target_delta[idx], dtype=np.float32)),
            "proposal_safe_target_action_local": torch.from_numpy(np.asarray(safe_target[idx], dtype=np.float32)),
            "current_pose_7d": torch.from_numpy(np.asarray(current_pose[idx], dtype=np.float32)),
            "target_pose_7d": torch.from_numpy(np.asarray(target_pose[idx], dtype=np.float32)),
            "yaw_augmentation_applied": torch.tensor(float(np.asarray(yaw_aug[idx])), dtype=torch.float32),
            "yaw_opportunity_label": torch.tensor(float(np.asarray(yaw_opp[idx])), dtype=torch.float32),
        }
