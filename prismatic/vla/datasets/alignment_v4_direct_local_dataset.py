"""Dataset for alignment_v4_direct_local_controller."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset


class AlignmentV4DirectLocalDataset(Dataset):
    """Compatibility dataset for V4 training and shadow audits.

    The loader accepts either a fresh V4 contract-matched npz or a compatible
    V3 from-scratch npz and normalizes the fields into the V4 training contract.
    """

    def __init__(self, npz_path: str, stage_bucket_filter: Optional[List[str]] = None):
        data = np.load(Path(npz_path), allow_pickle=True)
        self._data = {k: np.asarray(data[k]) for k in data.files}
        if "current_to_target_delta_local" in self._data:
            self.length = int(self._data["current_to_target_delta_local"].shape[0])
        elif "runtime_current_to_target_delta_local" in self._data:
            self.length = int(self._data["runtime_current_to_target_delta_local"].shape[0])
        else:
            raise KeyError("dataset missing current_to_target_delta_local/runtime_current_to_target_delta_local")

        if stage_bucket_filter is not None and "stage_bucket" in self._data:
            buckets = self._data.get("stage_bucket", np.array(["unknown"] * self.length))
            mask = np.array([str(b) in stage_bucket_filter for b in buckets], dtype=bool)
        else:
            mask = np.ones(self.length, dtype=bool)
        self._indices = np.where(mask)[0]

    def __len__(self) -> int:
        return int(self._indices.size)

    def _arr(self, i: int, key: str, dtype=np.float32, fallback=None):
        if key in self._data:
            return torch.from_numpy(np.asarray(self._data[key][i], dtype=dtype))
        if fallback is None:
            raise KeyError(key)
        return torch.from_numpy(np.asarray(fallback, dtype=dtype))

    def _scalar(self, i: int, key: str, default=0.0):
        if key not in self._data:
            return float(default)
        try:
            return float(np.asarray(self._data[key][i]).reshape(()))
        except Exception:
            return float(default)

    def _label_from_residual(self, residual6: np.ndarray, eps_pos: float = 1e-4, eps_yaw: float = 1e-4) -> int:
        pos_norm = float(np.linalg.norm(residual6[:3]))
        yaw_abs = float(abs(residual6[5]))
        if pos_norm <= eps_pos and yaw_abs <= eps_yaw:
            return 2  # hold
        return 1  # assist

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor | str]:
        i = int(self._indices[idx])

        def _fallback_shape(key: str, shape: tuple[int, ...], dtype=np.float32):
            if key in self._data:
                arr = np.asarray(self._data[key][i], dtype=dtype)
                return torch.from_numpy(arr)
            return torch.zeros(shape, dtype=torch.float32)

        front_rgb = _fallback_shape("front_rgb", (3, 96, 96))
        if front_rgb.ndim == 3 and front_rgb.shape[0] != 3 and front_rgb.shape[-1] == 3:
            front_rgb = front_rgb.permute(2, 0, 1).contiguous()
        if front_rgb.max() > 1.5:
            front_rgb = front_rgb / 255.0

        wrist_depth = _fallback_shape("wrist_depth", (1, 96, 96))
        if wrist_depth.ndim == 2:
            wrist_depth = wrist_depth.unsqueeze(0)
        elif wrist_depth.ndim == 3 and wrist_depth.shape[0] != 1:
            wrist_depth = wrist_depth[:1]

        force_history = _fallback_shape("force_history", (32, 6))
        if force_history.ndim != 2:
            force_history = force_history.reshape(-1, force_history.shape[-1]) if force_history.numel() else torch.zeros((32, 6), dtype=torch.float32)

        proprio = _fallback_shape("proprio", (15,))

        planner_action = None
        for key in ("planner_base_action_local", "planner_action_local", "planner_base_action_local_raw", "base_action"):
            if key in self._data:
                arr = np.asarray(self._data[key][i], dtype=np.float32).reshape(-1)
                planner_action = torch.from_numpy(arr[:6].copy())
                break
        if planner_action is None:
            planner_action = torch.zeros((6,), dtype=torch.float32)

        current_delta = None
        for key in ("current_to_target_delta_local", "runtime_current_to_target_delta_local", "teacher_current_to_target_delta_local"):
            if key in self._data:
                current_delta = np.asarray(self._data[key][i], dtype=np.float32).reshape(-1)[:6].copy()
                break
        if current_delta is None:
            current_delta = np.zeros((6,), dtype=np.float32)

        teacher_delta = None
        for key in ("teacher_current_to_target_delta_local", "privileged_current_to_target_delta_local", "target_residual_source_delta_local"):
            if key in self._data:
                teacher_delta = np.asarray(self._data[key][i], dtype=np.float32).reshape(-1)[:6].copy()
                break
        if teacher_delta is None:
            teacher_delta = current_delta.copy()

        residual = None
        if "alignment_v4_residual_target" in self._data:
            residual = np.asarray(self._data["alignment_v4_residual_target"][i], dtype=np.float32).reshape(-1)[:4].copy()
        elif "target_residual_local_4d" in self._data:
            residual = np.asarray(self._data["target_residual_local_4d"][i], dtype=np.float32).reshape(-1)[:4].copy()
        else:
            residual6 = current_delta - teacher_delta
            residual = np.asarray([residual6[0], residual6[1], residual6[2], residual6[5]], dtype=np.float32)

        if "alignment_v4_residual_target" in self._data:
            target_post = current_delta.copy()
            target_post[:3] -= np.asarray(residual[:3], dtype=np.float32)
            target_post[5] -= float(residual[3])
        else:
            target_post = np.asarray(
                [
                    self._scalar(i, "target_post_xy_error", abs(current_delta[0])),
                    self._scalar(i, "target_post_z_error", abs(current_delta[2])),
                    self._scalar(i, "target_post_yaw_error", abs(current_delta[5])),
                ],
                dtype=np.float32,
            )

        current_xy = float(np.linalg.norm(current_delta[:2]))
        current_z = float(abs(current_delta[2]))
        current_yaw = float(abs(current_delta[5]))
        target_post_xy = float(target_post[0]) if target_post.shape[0] >= 1 else current_xy
        target_post_z = float(target_post[1]) if target_post.shape[0] >= 2 else current_z
        target_post_yaw = float(target_post[2]) if target_post.shape[0] >= 3 else current_yaw

        target_improves_xy = float(target_post_xy < current_xy)
        target_improves_z = float(target_post_z < current_z)
        target_improves_yaw = float(target_post_yaw < current_yaw)
        target_noop = self._label_from_residual(np.asarray([residual[0], residual[1], residual[2], 0.0, 0.0, residual[3]], dtype=np.float32))

        closeability = self._scalar(i, "alignment_v4_closeability_label", self._scalar(i, "closeability_label", 0.0))
        if closeability <= 0.0:
            closeability = float(target_improves_xy or target_improves_z or target_improves_yaw)

        if target_noop == 2 and closeability > 0.5:
            policy_mode = 2  # hold
        elif target_noop == 2:
            policy_mode = 0  # noop
        else:
            policy_mode = 1  # assist

        out: Dict[str, torch.Tensor | str] = {
            "row_index": torch.tensor(i, dtype=torch.long),
            "front_rgb": front_rgb.float(),
            "wrist_depth": wrist_depth.float(),
            "force_history": force_history.float(),
            "proprio": proprio.float(),
            "planner_action_local": planner_action.float(),
            "current_to_target_delta_local": torch.from_numpy(current_delta[:6]).float(),
            "runtime_current_to_target_delta_local": torch.from_numpy(
                np.asarray(self._data.get("runtime_current_to_target_delta_local", self._data.get("current_to_target_delta_local"))[i], dtype=np.float32).reshape(-1)[:6].copy()
            ).float()
            if "runtime_current_to_target_delta_local" in self._data or "current_to_target_delta_local" in self._data
            else torch.from_numpy(current_delta[:6]).float(),
            "teacher_current_to_target_delta_local": torch.from_numpy(teacher_delta[:6]).float(),
            "target_residual_local_4d": torch.from_numpy(residual[:4]).float(),
            "target_residual_local_6d": torch.from_numpy(np.asarray([residual[0], residual[1], residual[2], 0.0, 0.0, residual[3]], dtype=np.float32)).float(),
            "target_post_xy_error": torch.tensor(target_post_xy, dtype=torch.float32),
            "target_post_z_error": torch.tensor(target_post_z, dtype=torch.float32),
            "target_post_yaw_error": torch.tensor(target_post_yaw, dtype=torch.float32),
            "current_xy_error": torch.tensor(current_xy, dtype=torch.float32),
            "current_z_error": torch.tensor(current_z, dtype=torch.float32),
            "current_yaw_error": torch.tensor(current_yaw, dtype=torch.float32),
            "target_improves_xy": torch.tensor(target_improves_xy, dtype=torch.float32),
            "target_improves_z": torch.tensor(target_improves_z, dtype=torch.float32),
            "target_improves_yaw": torch.tensor(target_improves_yaw, dtype=torch.float32),
            "invalid_risk_proxy": torch.tensor(self._scalar(i, "invalid_risk_proxy", 0.0), dtype=torch.float32),
            "overshoot_proxy": torch.tensor(self._scalar(i, "overshoot_proxy", 0.0), dtype=torch.float32),
            "policy_mode_label": torch.tensor(policy_mode, dtype=torch.long),
            "stage_bucket": str(self._data["stage_bucket"][i]) if "stage_bucket" in self._data else "unknown",
            "sample_weight": torch.tensor(self._scalar(i, "sample_weight", 1.0), dtype=torch.float32),
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
            "episode_index",
            "step_index",
        ):
            if key not in self._data:
                continue
            val = self._data[key][i]
            if key == "gripper_context":
                out[key] = torch.from_numpy(np.asarray(val, dtype=np.float32))
            elif key in ("runtime_target_delta_source", "runtime_target_delta_context_mode"):
                out[key] = str(val)
            elif key in ("substage_id", "contact_state", "stage_target_mode", "episode_index", "step_index"):
                out[key] = torch.tensor(int(val), dtype=torch.long)
            else:
                out[key] = torch.tensor(float(val), dtype=torch.float32)
        return out
