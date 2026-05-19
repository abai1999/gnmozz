"""
residual_rlbench_dataset.py

Map-style PyTorch dataset for stage-refiner training.

This loader is backward compatible with the old residual shards, while exposing
the new planner-conditioned readiness / alignment fields used by NFCR v2.
"""

import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from torch.utils.data import Dataset


class ResidualRLBenchDataset(Dataset):
    """Dataset for residual controller training from pre-collected .npz shards."""

    DEFAULT_KEYS = (
        "wrist_depth",
        "ft_hist",
        "proprio",
        "base_action",
        "gripper_context",
        "interaction_role",
        "step_idx",
        "delta_target",
        "delta_align_target",
        "delta_basin_target",
        "contact_mask",
        "phase_label",
        "phase_id",
        "phase_age",
        "steps_since_last_replan",
        "stage_role",
        "failure_mode",
        "transition_flag",
        "subgoal_progress",
        "rollout_gripper_open",
        "depth_proximity",
        "planner_close_intent",
        "planner_close_intent_strength",
        "readiness_label",
        "basin_positive",
        "basin_distance",
        "hold_label",
        "negative_reason",
        "frames_to_expert_close",
        "frames_to_reference_trigger",
        "post_close_stability_proxy",
        "grasp_lift_proxy",
        "reopen_within_horizon",
        "reopen_after_trigger",
        "no_progress_after_trigger",
        "invalid_after_trigger",
        "gripper_state_target",
        "ready_to_close",
        "planner_close_too_early",
        "expert_hold_after_close",
    )

    def __init__(
        self,
        data_dir: str,
        oversample_contact: int = 5,
        oversample_pre_contact: int = 3,
        oversample_jam: int = 7,
        stage_role_filter: Optional[str] = None,
    ):
        self.data_dir = Path(data_dir)
        meta_path = self.data_dir / "residual_meta.json"
        self.meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}

        shard_files = sorted(self.data_dir.glob("residual_shard_*.npz"))
        assert shard_files, f"No residual shards found in {self.data_dir}"

        raw = {key: [] for key in self.DEFAULT_KEYS}
        for sf in shard_files:
            shard = np.load(sf)
            n = int(shard["base_action"].shape[0])
            for key in self.DEFAULT_KEYS:
                raw[key].append(self._load_key(shard, key, n))

        self._data = {key: np.concatenate(values, axis=0) for key, values in raw.items()}
        self.total_raw = int(self._data["base_action"].shape[0])

        contact_mask = self._data["contact_mask"]
        phase_id = self._data["phase_id"]
        stage_role = self._data["stage_role"]
        readiness_label = self._data["readiness_label"]
        hold_label = self._data["hold_label"]
        planner_close_intent = self._data["planner_close_intent"]
        negative_reason = self._data["negative_reason"]

        indices = []
        for i in range(self.total_raw):
            if stage_role_filter == "align" and int(stage_role[i]) != 0:
                continue
            if stage_role_filter == "contact" and int(stage_role[i]) != 1:
                continue

            repeat = 1
            if int(phase_id[i]) == 3:
                repeat = oversample_jam
            elif int(contact_mask[i]) == 2:
                repeat = oversample_contact
            elif int(contact_mask[i]) == 1:
                repeat = oversample_pre_contact

            if readiness_label[i] > 0.5:
                repeat = max(repeat, oversample_pre_contact + 4)
            if planner_close_intent[i] > 0.5 and readiness_label[i] < 0.5:
                repeat = max(repeat, oversample_pre_contact + 2)
            if hold_label[i] > 0.5:
                repeat = max(repeat, oversample_pre_contact + 2)
            if int(negative_reason[i]) == 0:
                repeat = max(repeat, oversample_pre_contact + 3)
            elif int(negative_reason[i]) in (1, 3):
                repeat = max(repeat, oversample_pre_contact + 2)
            elif int(negative_reason[i]) == 2:
                repeat = max(repeat, oversample_pre_contact + 1)
            elif int(negative_reason[i]) == 4:
                repeat = max(repeat, oversample_pre_contact + 3)
            indices.extend([i] * repeat)

        self._indices = np.asarray(indices, dtype=np.int64)

        self.n_free = int((contact_mask == 0).sum())
        self.n_pre_contact = int((contact_mask == 1).sum())
        self.n_contact = int((contact_mask == 2).sum())
        self.phase_counts = self._count_values(phase_id)
        self.stage_role_counts = self._count_values(stage_role)
        self.failure_counts = self._count_values(self._data["failure_mode"])
        self.readiness_counts = self._count_values(readiness_label.astype(np.int64))
        self.hold_counts = self._count_values(hold_label.astype(np.int64))
        self.negative_reason_counts = self._count_values(negative_reason[negative_reason >= 0].astype(np.int64))
        self.planner_close_intent_counts = self._count_values(planner_close_intent.astype(np.int64))
        self.gripper_state_counts = self._count_values(self._data["gripper_state_target"][self._data["gripper_state_target"] >= 0])

    @staticmethod
    def _count_values(values: np.ndarray) -> Dict[int, int]:
        if values.size == 0:
            return {}
        return {int(v): int((values == v).sum()) for v in np.unique(values)}

    @staticmethod
    def _zeros(shape, dtype):
        return np.zeros(shape, dtype=dtype)

    def _load_key(self, shard, key: str, n: int):
        if key in shard:
            return shard[key]
        if key == "basin_positive":
            if "readiness_label" in shard:
                return shard["readiness_label"]
            return np.full((n,), -1.0, dtype=np.float32)
        if key == "frames_to_reference_trigger":
            if "frames_to_expert_close" in shard:
                return shard["frames_to_expert_close"]
            return np.full((n,), -1, dtype=np.int64)
        if key == "reopen_after_trigger":
            if "reopen_within_horizon" in shard:
                return shard["reopen_within_horizon"]
            return np.full((n,), -1.0, dtype=np.float32)

        ref_idx = shard["step_idx"] if "step_idx" in shard else np.arange(n, dtype=np.int64)
        if key == "delta_align_target":
            return shard["delta_target"] if "delta_target" in shard else self._zeros((n, 6), np.float32)
        if key == "delta_basin_target":
            if "delta_basin_target" in shard:
                return shard["delta_basin_target"]
            if "delta_align_target" in shard:
                return shard["delta_align_target"]
            return shard["delta_target"] if "delta_target" in shard else self._zeros((n, 6), np.float32)
        if key == "gripper_context":
            return self._zeros((n, 3), np.float32)
        if key == "interaction_role":
            return self._zeros((n,), np.int64)
        if key in ("phase_age", "steps_since_last_replan", "subgoal_progress"):
            return self._zeros((n,), np.float32)
        if key in ("failure_mode", "transition_flag", "negative_reason"):
            fill = -1 if key == "negative_reason" else 0
            return np.full((n,), fill, dtype=np.int64)
        if key in ("rollout_gripper_open", "depth_proximity", "planner_close_intent_strength", "basin_distance"):
            return np.full((n,), -1.0, dtype=np.float32)
        if key in ("planner_close_intent", "readiness_label", "basin_positive", "hold_label", "ready_to_close", "planner_close_too_early", "expert_hold_after_close", "post_close_stability_proxy", "grasp_lift_proxy", "reopen_within_horizon", "reopen_after_trigger", "no_progress_after_trigger", "invalid_after_trigger"):
            return np.full((n,), -1.0, dtype=np.float32)
        if key in ("frames_to_expert_close", "frames_to_reference_trigger", "gripper_state_target"):
            return np.full((n,), -1, dtype=np.int64)
        if key == "contact_mask":
            return shard["phase_label"] if "phase_label" in shard else self._zeros((n,), np.int64)
        if key == "phase_label":
            return shard["contact_mask"] if "contact_mask" in shard else self._zeros((n,), np.int64)
        if key == "phase_id":
            if "phase_label" in shard:
                phase_label = shard["phase_label"]
            elif "contact_mask" in shard:
                phase_label = shard["contact_mask"]
            else:
                phase_label = self._zeros((n,), np.int64)
            mapped = np.zeros_like(phase_label, dtype=np.int64)
            mapped[phase_label == 1] = 1
            mapped[np.isin(phase_label, [2, 3])] = 2
            mapped[phase_label == 4] = 3
            return mapped
        if key == "stage_role":
            phase_id = self._load_key(shard, "phase_id", n)
            mapped = np.zeros_like(phase_id, dtype=np.int64)
            mapped[np.isin(phase_id, [2, 3])] = 1
            return mapped

        raise KeyError(f"Missing key '{key}' in shard")

    def __len__(self) -> int:
        return int(self._indices.shape[0])

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        i = int(self._indices[idx])
        sample = {
            "wrist_depth": torch.from_numpy(self._data["wrist_depth"][i]),
            "ft_hist": torch.from_numpy(self._data["ft_hist"][i]),
            "proprio": torch.from_numpy(self._data["proprio"][i]),
            "base_action": torch.from_numpy(self._data["base_action"][i]),
            "gripper_context": torch.from_numpy(self._data["gripper_context"][i]),
            "interaction_role": torch.tensor(self._data["interaction_role"][i], dtype=torch.long),
            "step_idx": torch.tensor(self._data["step_idx"][i], dtype=torch.long),
            "delta_target": torch.from_numpy(self._data["delta_target"][i]),
            "delta_align_target": torch.from_numpy(self._data["delta_align_target"][i]),
            "delta_basin_target": torch.from_numpy(self._data["delta_basin_target"][i]),
            "contact_mask": torch.tensor(self._data["contact_mask"][i], dtype=torch.long),
            "phase_label": torch.tensor(self._data["phase_label"][i], dtype=torch.long),
            "phase_id": torch.tensor(self._data["phase_id"][i], dtype=torch.long),
            "phase_age": torch.tensor(self._data["phase_age"][i], dtype=torch.float32),
            "steps_since_last_replan": torch.tensor(self._data["steps_since_last_replan"][i], dtype=torch.float32),
            "stage_role": torch.tensor(self._data["stage_role"][i], dtype=torch.long),
            "failure_mode": torch.tensor(self._data["failure_mode"][i], dtype=torch.long),
            "transition_flag": torch.tensor(self._data["transition_flag"][i], dtype=torch.long),
            "subgoal_progress": torch.tensor(self._data["subgoal_progress"][i], dtype=torch.float32),
            "rollout_gripper_open": torch.tensor(self._data["rollout_gripper_open"][i], dtype=torch.float32),
            "depth_proximity": torch.tensor(self._data["depth_proximity"][i], dtype=torch.float32),
            "planner_close_intent": torch.tensor(self._data["planner_close_intent"][i], dtype=torch.float32),
            "planner_close_intent_strength": torch.tensor(self._data["planner_close_intent_strength"][i], dtype=torch.float32),
            "readiness_label": torch.tensor(self._data["readiness_label"][i], dtype=torch.float32),
            "basin_positive": torch.tensor(self._data["basin_positive"][i], dtype=torch.float32),
            "basin_distance": torch.tensor(self._data["basin_distance"][i], dtype=torch.float32),
            "hold_label": torch.tensor(self._data["hold_label"][i], dtype=torch.float32),
            "negative_reason": torch.tensor(self._data["negative_reason"][i], dtype=torch.long),
            "frames_to_expert_close": torch.tensor(self._data["frames_to_expert_close"][i], dtype=torch.long),
            "frames_to_reference_trigger": torch.tensor(self._data["frames_to_reference_trigger"][i], dtype=torch.long),
            "post_close_stability_proxy": torch.tensor(self._data["post_close_stability_proxy"][i], dtype=torch.float32),
            "grasp_lift_proxy": torch.tensor(self._data["grasp_lift_proxy"][i], dtype=torch.float32),
            "reopen_within_horizon": torch.tensor(self._data["reopen_within_horizon"][i], dtype=torch.float32),
            "reopen_after_trigger": torch.tensor(self._data["reopen_after_trigger"][i], dtype=torch.float32),
            "no_progress_after_trigger": torch.tensor(self._data["no_progress_after_trigger"][i], dtype=torch.float32),
            "invalid_after_trigger": torch.tensor(self._data["invalid_after_trigger"][i], dtype=torch.float32),
            "gripper_state_target": torch.tensor(self._data["gripper_state_target"][i], dtype=torch.long),
            "ready_to_close": torch.tensor(self._data["ready_to_close"][i], dtype=torch.float32),
            "planner_close_too_early": torch.tensor(self._data["planner_close_too_early"][i], dtype=torch.float32),
            "expert_hold_after_close": torch.tensor(self._data["expert_hold_after_close"][i], dtype=torch.float32),
        }
        return sample

    def get_summary(self) -> dict:
        deltas = np.asarray(self._data["delta_align_target"], dtype=np.float32)
        delta_xy = np.linalg.norm(deltas[:, :2], axis=-1) if deltas.size else np.zeros((0,), dtype=np.float32)
        delta_yaw = np.abs(deltas[:, 5]) if deltas.size else np.zeros((0,), dtype=np.float32)
        basin_distance = np.asarray(self._data["basin_distance"], dtype=np.float32)
        valid_basin = basin_distance[basin_distance >= 0]
        return {
            "total_raw_samples": self.total_raw,
            "total_oversampled": len(self._indices),
            "dataset_view": self.meta.get("dataset_view"),
            "n_free": self.n_free,
            "n_pre_contact": self.n_pre_contact,
            "n_contact": self.n_contact,
            "phase_counts": self.phase_counts,
            "stage_role_counts": self.stage_role_counts,
            "failure_counts": self.failure_counts,
            "readiness_counts": self.readiness_counts,
            "basin_positive_counts": self.readiness_counts,
            "hold_counts": self.hold_counts,
            "negative_reason_counts": self.negative_reason_counts,
            "planner_close_intent_counts": self.planner_close_intent_counts,
            "gripper_state_counts": self.gripper_state_counts,
            "delta_xy_p50": float(np.percentile(delta_xy, 50)) if delta_xy.size else 0.0,
            "delta_xy_p95": float(np.percentile(delta_xy, 95)) if delta_xy.size else 0.0,
            "delta_yaw_p50": float(np.percentile(delta_yaw, 50)) if delta_yaw.size else 0.0,
            "delta_yaw_p95": float(np.percentile(delta_yaw, 95)) if delta_yaw.size else 0.0,
            "basin_distance_p50": float(np.percentile(valid_basin, 50)) if valid_basin.size else -1.0,
            "basin_distance_p95": float(np.percentile(valid_basin, 95)) if valid_basin.size else -1.0,
        }
