"""
dagger_rlbench_dataset.py

Map-style dataset for planner DAgger rollout samples collected from planner-state rollouts.
"""

import json
import random
from pathlib import Path
from typing import Any, Dict, Optional, Type

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

from prismatic.models.backbones.llm.prompting import PromptBuilder, QwenPromptBuilder
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.constants import FORCE_DIM, FORCE_HISTORY_LEN, IGNORE_INDEX, NUM_ACTIONS_CHUNK, NUM_TOKENS


class DaggerRLBenchDataset(Dataset):
    """Planner-state DAgger samples saved as compressed shards."""

    def __init__(
        self,
        data_dir: str,
        image_transform,
        action_tokenizer: ActionTokenizer,
        tokenizer: PreTrainedTokenizerBase,
        prompt_builder_fn: Type[PromptBuilder],
        use_depth: bool = False,
        use_force: bool = False,
        image_aug: bool = False,
        oversample_align: int = 3,
        oversample_interact: int = 2,
        oversample_transition: int = 4,
        oversample_failure: int = 5,
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.image_transform = image_transform
        self.action_tokenizer = action_tokenizer
        self.tokenizer = tokenizer
        self.prompt_builder_fn = prompt_builder_fn
        self.use_depth = use_depth
        self.use_force = use_force
        self.image_aug = image_aug
        self.oversample_align = oversample_align
        self.oversample_interact = oversample_interact
        self.oversample_transition = oversample_transition
        self.oversample_failure = oversample_failure

        shard_files = sorted(self.data_dir.glob("dagger_shard_*.npz"))
        assert shard_files, f"No dagger shards found in {self.data_dir}"

        all_data = {
            "front_rgb": [],
            "wrist_rgb": [],
            "wrist_depth": [],
            "force_history": [],
            "proprio": [],
            "action_chunk": [],
            "language": [],
            "phase_id": [],
            "step_idx": [],
            "event_code": [],
            "transition_flag": [],
            "failure_mode": [],
        }
        for sf in shard_files:
            shard = np.load(sf, allow_pickle=True)
            for key in all_data:
                if key in shard:
                    all_data[key].append(shard[key])
                elif key in ("transition_flag", "failure_mode"):
                    ref = shard["phase_id"]
                    all_data[key].append(np.zeros((ref.shape[0],), dtype=np.int64))
                elif key == "wrist_depth":
                    ref = shard["front_rgb"]
                    all_data[key].append(np.zeros((ref.shape[0], 1, 96, 96), dtype=np.float32))
                elif key == "force_history":
                    ref = shard["front_rgb"]
                    all_data[key].append(np.zeros((ref.shape[0], FORCE_HISTORY_LEN, FORCE_DIM), dtype=np.float32))
                else:
                    raise KeyError(f"Missing key '{key}' in {sf}")

        for key in all_data:
            all_data[key] = np.concatenate(all_data[key], axis=0)
        self._data = all_data
        self.phase_counts = {
            int(v): int((self._data["phase_id"] == v).sum()) for v in np.unique(self._data["phase_id"])
        }
        self.event_counts = {
            int(v): int((self._data["event_code"] == v).sum()) for v in np.unique(self._data["event_code"])
        }
        self.failure_counts = {
            int(v): int((self._data["failure_mode"] == v).sum()) for v in np.unique(self._data["failure_mode"])
        }
        indices = []
        for i in range(len(self._data["action_chunk"])):
            repeat = 1
            phase_id = int(self._data["phase_id"][i])
            if phase_id == 1:
                repeat = max(repeat, self.oversample_align)
            elif phase_id >= 2:
                repeat = max(repeat, self.oversample_interact)
            if int(self._data["transition_flag"][i]) > 0 or int(self._data["event_code"][i]) == 2:
                repeat = max(repeat, self.oversample_transition)
            if int(self._data["failure_mode"][i]) > 0 or int(self._data["event_code"][i]) == 3:
                repeat = max(repeat, self.oversample_failure)
            indices.extend([i] * repeat)
        self._indices = np.asarray(indices, dtype=np.int64)
        self._len = len(self._indices)

        meta_path = self.data_dir / "dagger_meta.json"
        self.meta = {}
        if meta_path.exists():
            with open(meta_path) as f:
                self.meta = json.load(f)

    def __len__(self):
        return self._len

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        real_idx = int(self._indices[idx])
        front_img = Image.fromarray(self._data["front_rgb"][real_idx].astype(np.uint8)).convert("RGB")
        wrist_img = Image.fromarray(self._data["wrist_rgb"][real_idx].astype(np.uint8)).convert("RGB")
        pixel_values = self.image_transform(front_img)
        pixel_values_wrist = self.image_transform(wrist_img)
        proprio = self._data["proprio"][real_idx].astype(np.float32)
        action_chunk = self._data["action_chunk"][real_idx].astype(np.float32)

        prompt_builder = QwenPromptBuilder("openvla")
        lang = str(self._data["language"][real_idx])
        conversation = [
            {"from": "human", "value": f"What action should the robot take to {lang}?"},
            {"from": "gpt", "value": ""},
        ]
        for turn in conversation:
            prompt_builder.add_turn(turn["from"], turn["value"])
        input_ids = self.tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids
        if len(input_ids) >= 3:
            del input_ids[-1]
            del input_ids[-1]
            del input_ids[-1]

        action_tokens = self.action_tokenizer(action_chunk.flatten(), use_minivlm=True)
        if NUM_TOKENS < len(action_tokens):
            action_tokens = action_tokens[:NUM_TOKENS]
        else:
            remaining = NUM_TOKENS - len(action_tokens)
            if remaining > 0 and len(action_tokens) > 0:
                action_tokens = action_tokens + random.choices(action_tokens, k=remaining)
        input_ids = input_ids + action_tokens
        labels = list(input_ids)
        action_chunk_len = len(action_tokens)
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        labels = torch.tensor(labels, dtype=torch.long)
        labels[: -(action_chunk_len + 1)] = IGNORE_INDEX

        item = {
            "pixel_values": pixel_values,
            "pixel_values_wrist": pixel_values_wrist,
            "input_ids": input_ids,
            "labels": labels,
            "actions": action_chunk.astype(np.float32),
            "proprio": torch.from_numpy(proprio),
            "dataset_name": "dagger_rlbench",
        }
        if self.use_depth:
            item["wrist_depth"] = torch.from_numpy(self._data["wrist_depth"][real_idx].astype(np.float32))
        if self.use_force:
            item["force_history"] = torch.from_numpy(self._data["force_history"][real_idx].astype(np.float32))
        return item

    def get_summary(self) -> Dict[str, Any]:
        return {
            "total_raw_samples": int(len(self._data["action_chunk"])),
            "total_oversampled": int(self._len),
            "phase_counts": self.phase_counts,
            "event_counts": self.event_counts,
            "failure_counts": self.failure_counts,
            "oversample_align": self.oversample_align,
            "oversample_interact": self.oversample_interact,
            "oversample_transition": self.oversample_transition,
            "oversample_failure": self.oversample_failure,
        }
