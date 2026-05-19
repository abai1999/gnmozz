"""
rlbench_dataset.py

Map-style PyTorch Dataset for loading RLBench episodes with depth, force, and phase annotations.
Compatible with PaddedCollatorForActionPrediction.
"""

import json
import os
import pickle
import random
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

import numpy as np
import torch
from PIL import Image
from scipy.spatial.transform import Rotation
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

from prismatic.models.backbones.llm.prompting import PromptBuilder, QwenPromptBuilder
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.constants import (
    ACTION_DIM,
    FORCE_DIM,
    FORCE_HISTORY_LEN,
    IGNORE_INDEX,
    NUM_ACTIONS_CHUNK,
    NUM_TOKENS,
)


class RLBenchDataset(Dataset):
    """Map-style dataset that loads RLBench episodes with optional depth, force, and phase modalities."""

    def __init__(
        self,
        data_root: str,
        task_name: str,
        image_transform: Callable,
        action_tokenizer: ActionTokenizer,
        tokenizer: PreTrainedTokenizerBase,
        prompt_builder_fn: Type[PromptBuilder],
        use_depth: bool = True,
        use_force: bool = True,
        force_history_len: int = FORCE_HISTORY_LEN,
        image_aug: bool = False,
    ) -> None:
        super().__init__()
        self.data_root = Path(data_root)
        self.task_name = task_name
        self.image_transform = image_transform
        self.action_tokenizer = action_tokenizer
        self.tokenizer = tokenizer
        self.prompt_builder_fn = prompt_builder_fn
        self.use_depth = use_depth
        self.use_force = use_force
        self.force_history_len = force_history_len
        self.image_aug = image_aug

        # Resolve episode directory
        self.episodes_dir = self.data_root / task_name / "train" / "episodes"
        assert self.episodes_dir.exists(), f"Episodes dir not found: {self.episodes_dir}"

        # Load dataset statistics for normalization
        stats_path = self.episodes_dir / "dataset_statistics.json"
        if not stats_path.exists():
            stats_path = self.data_root / task_name / "dataset_statistics.json"
        assert stats_path.exists(), f"dataset_statistics.json not found for {task_name}"
        with open(stats_path) as f:
            stats = json.load(f)
        self.stats = stats.get("rlbench", stats)
        self.action_q01 = np.array(self.stats["action"]["q01"], dtype=np.float32)
        self.action_q99 = np.array(self.stats["action"]["q99"], dtype=np.float32)
        if "force" in self.stats:
            self.force_mean = np.array(self.stats["force"]["mean"], dtype=np.float32)
            self.force_std = np.array(self.stats["force"]["std"], dtype=np.float32)
            self.force_std = np.maximum(self.force_std, 1e-6)  # avoid div-by-zero
        else:
            self.force_mean = np.zeros(FORCE_DIM, dtype=np.float32)
            self.force_std = np.ones(FORCE_DIM, dtype=np.float32)

        # Build episode index: list of (episode_dir, num_frames)
        self.episode_index: List[Tuple[Path, int]] = []
        self._frame_offsets: List[int] = []  # cumulative frame count for global indexing

        ep_dirs = sorted(
            [d for d in os.listdir(self.episodes_dir) if d.startswith("episode")],
            key=lambda x: int(x.replace("episode", "")),
        )
        cumulative = 0
        for ep_name in ep_dirs:
            ep_path = self.episodes_dir / ep_name
            npz_path = ep_path / "model_inputs.npz"
            if not npz_path.exists():
                continue
            npz = np.load(npz_path)
            # Validate episode has required data
            if "action_targets" not in npz or "proprio" not in npz:
                continue
            n_frames = npz["action_targets"].shape[0]
            # Need at least NUM_ACTIONS_CHUNK frames to form a complete action chunk
            if n_frames < NUM_ACTIONS_CHUNK:
                continue
            self.episode_index.append((ep_path, n_frames))
            self._frame_offsets.append(cumulative)
            cumulative += n_frames

        self._total_frames = cumulative

        # Cache language descriptions per episode (loaded lazily)
        self._lang_cache: Dict[str, List[str]] = {}

        # Expose dataset_statistics in the format expected by save_dataset_statistics()
        # Format: {"rlbench": {"action": {...}, "proprio": {...}, "force": {...}}}
        self.dataset_statistics = stats

    def __len__(self) -> int:
        return self._total_frames

    def _global_to_local(self, idx: int) -> Tuple[int, int]:
        """Convert global frame index to (episode_idx, frame_idx)."""
        # Binary search for the episode
        lo, hi = 0, len(self._frame_offsets) - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if self._frame_offsets[mid] <= idx:
                lo = mid
            else:
                hi = mid - 1
        frame_idx = idx - self._frame_offsets[lo]
        return lo, frame_idx

    def _load_language(self, ep_path: Path) -> str:
        """Load language description for an episode."""
        key = str(ep_path)
        if key not in self._lang_cache:
            desc_path = ep_path / "variation_descriptions.pkl"
            if desc_path.exists():
                with open(desc_path, "rb") as f:
                    descs = pickle.load(f)
                if isinstance(descs, list):
                    self._lang_cache[key] = [d.lower().strip() for d in descs]
                else:
                    self._lang_cache[key] = [str(descs).lower().strip()]
            else:
                self._lang_cache[key] = [self.task_name.replace("_", " ")]
        return random.choice(self._lang_cache[key])

    def _load_image(self, ep_path: Path, cam: str, frame_idx: int) -> Image.Image:
        """Load an RGB image from the specified camera."""
        img_path = ep_path / cam / f"{frame_idx}.png"
        return Image.open(img_path).convert("RGB")

    def _load_depth(self, ep_path: Path, frame_idx: int) -> np.ndarray:
        """Load wrist depth image, normalize to [0, 1], return as (1, H, W) float32."""
        depth_path = ep_path / "wrist_depth" / f"{frame_idx}.png"
        depth_img = Image.open(depth_path)
        depth = np.array(depth_img, dtype=np.float32)
        # RLBench depth is saved as float encoded in RGB; handle both cases
        if depth.ndim == 3:
            # Use first channel if multi-channel
            depth = depth[:, :, 0]
        # Clip to max 2.0m and normalize to [0, 1]
        depth = np.clip(depth / 255.0, 0.0, 1.0)
        return depth

    def _get_action_7d(self, npz_data: dict, frame_idx: int) -> np.ndarray:
        """Extract 7D action chunk starting at frame_idx (delta_pos + delta_rotvec + next_gripper)."""
        at = npz_data["action_targets"]
        T = at.shape[0]
        action_dim = at.shape[1]

        # Build chunk of NUM_ACTIONS_CHUNK actions
        chunk = np.zeros((NUM_ACTIONS_CHUNK, ACTION_DIM), dtype=np.float32)
        for i in range(NUM_ACTIONS_CHUNK):
            t = min(frame_idx + i, T - 1)
            if action_dim == ACTION_DIM:
                chunk[i] = at[t]
            else:
                # For 10D or other formats: reconstruct 7D from gripper_pose + gripper_open
                gp = npz_data["gripper_pose"]
                go = npz_data["gripper_open"]
                if t < T - 1:
                    delta_pos = gp[t + 1, :3] - gp[t, :3]
                    r0 = Rotation.from_quat(gp[t, 3:7])
                    r1 = Rotation.from_quat(gp[t + 1, 3:7])
                    delta_rv = (r1 * r0.inv()).as_rotvec()
                    gripper = go[t + 1, 0]
                else:
                    delta_pos = np.zeros(3, dtype=np.float32)
                    delta_rv = np.zeros(3, dtype=np.float32)
                    gripper = go[t, 0]
                chunk[i] = np.concatenate([delta_pos, delta_rv, [gripper]])
        return chunk

    def _normalize_actions(self, actions: np.ndarray) -> np.ndarray:
        """Normalize actions to [-1, 1] using q01/q99 bounds."""
        q01 = self.action_q01
        q99 = self.action_q99
        mask = (q99 - q01) > 1e-8
        normalized = np.zeros_like(actions)
        normalized[:, mask] = 2.0 * (actions[:, mask] - q01[mask]) / (q99[mask] - q01[mask]) - 1.0
        normalized = np.clip(normalized, -1.0, 1.0)
        return normalized

    def _get_force_history(self, npz_data: dict, frame_idx: int) -> np.ndarray:
        """Get force history window ending at frame_idx, z-score normalized. Shape: (force_history_len, FORCE_DIM)."""
        forces = npz_data["gripper_touch_forces"]  # (T, 6)
        T = forces.shape[0]
        history = np.zeros((self.force_history_len, FORCE_DIM), dtype=np.float32)
        for i in range(self.force_history_len):
            t = frame_idx - (self.force_history_len - 1 - i)
            t = max(0, min(t, T - 1))
            history[i] = forces[t]
        # Z-score normalization
        history = (history - self.force_mean) / self.force_std
        return history

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ep_idx, frame_idx = self._global_to_local(idx)
        ep_path, n_frames = self.episode_index[ep_idx]

        # Load npz data
        npz_data = dict(np.load(ep_path / "model_inputs.npz"))

        # --- Language ---
        lang = self._load_language(ep_path)

        # --- Images ---
        front_img = self._load_image(ep_path, "front_rgb", frame_idx)
        wrist_img = self._load_image(ep_path, "wrist_rgb", frame_idx)
        pixel_values = self.image_transform(front_img)
        pixel_values_wrist = self.image_transform(wrist_img)

        # --- Proprio (15D) ---
        proprio = npz_data["proprio"][frame_idx].astype(np.float32)  # (15,)

        # --- Actions (7D x NUM_ACTIONS_CHUNK) ---
        action_chunk = self._get_action_7d(npz_data, frame_idx)
        action_chunk_normalized = self._normalize_actions(action_chunk)

        # --- Tokenize prompt ---
        prompt_builder = QwenPromptBuilder("openvla")
        conversation = [
            {"from": "human", "value": f"What action should the robot take to {lang}?"},
            {"from": "gpt", "value": ""},
        ]
        for turn in conversation:
            prompt_builder.add_turn(turn["from"], turn["value"])

        input_ids = self.tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids
        # Remove trailing tokens (empty assistant response tokens)
        if len(input_ids) >= 3:
            del input_ids[-3]
            del input_ids[-2]
            del input_ids[-1]

        # Tokenize action chunk into discrete bins
        action_tokens = self.action_tokenizer(action_chunk_normalized.flatten(), use_minivlm=True)
        if NUM_TOKENS < len(action_tokens):
            action_token_ids = action_tokens[:NUM_TOKENS]
        else:
            remaining = NUM_TOKENS - len(action_tokens)
            action_token_ids = action_tokens + random.choices(action_tokens, k=remaining)

        input_ids = input_ids + action_token_ids
        labels = list(input_ids)
        action_chunk_len = NUM_TOKENS

        input_ids = torch.tensor(input_ids, dtype=torch.long)
        labels = torch.tensor(labels, dtype=torch.long)
        # Mask everything before action tokens
        labels[:-(action_chunk_len + 1)] = IGNORE_INDEX

        # --- Build return dict ---
        return_dict: Dict[str, Any] = dict(
            pixel_values=pixel_values,
            pixel_values_wrist=pixel_values_wrist,
            input_ids=input_ids,
            labels=labels,
            dataset_name="rlbench",
            actions=action_chunk_normalized,  # (NUM_ACTIONS_CHUNK, ACTION_DIM)
            proprio=proprio,  # (15,)
        )

        # --- Depth (optional) ---
        if self.use_depth:
            depth = self._load_depth(ep_path, frame_idx)  # (H, W) in [0, 1]
            # Resize to 224x224 to match the visual encoder
            depth_pil = Image.fromarray((depth * 255).astype(np.uint8), mode="L")
            depth_pil = depth_pil.resize((224, 224), Image.BILINEAR)
            depth_tensor = torch.from_numpy(np.array(depth_pil, dtype=np.float32) / 255.0).unsqueeze(0)  # (1, 224, 224)
            return_dict["wrist_depth"] = depth_tensor

        # --- Force history (optional) ---
        if self.use_force:
            force_history = self._get_force_history(npz_data, frame_idx)  # (force_history_len, FORCE_DIM)
            return_dict["force_history"] = torch.from_numpy(force_history)

        # --- Phase ID (optional, only for enriched episodes) ---
        phase_path = ep_path / "phase_ids.npy"
        if phase_path.exists():
            phase_ids = np.load(phase_path)
            return_dict["phase_id"] = int(phase_ids[frame_idx])

        return return_dict
