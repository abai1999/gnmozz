"""
data_utils.py

General utilities and classes for facilitating data loading and collation.
"""

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Sequence, Tuple

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

# HuggingFace Default / LLaMa-2 IGNORE_INDEX (for labels)
IGNORE_INDEX = -100


def save_dataset_statistics(dataset_statistics: Dict, run_dir: Path) -> None:
    """Saves a `dataset_statistics.json` file for action/proprio denormalization at inference."""
    out_path = run_dir / "dataset_statistics.json"
    for _, stats in dataset_statistics.items():
        for k in stats.get("action", {}).keys():
            if isinstance(stats["action"][k], np.ndarray):
                stats["action"][k] = stats["action"][k].tolist()
        if "proprio" in stats:
            for k in stats["proprio"].keys():
                if isinstance(stats["proprio"][k], np.ndarray):
                    stats["proprio"][k] = stats["proprio"][k].tolist()
        if "force" in stats:
            for k in stats["force"].keys():
                if isinstance(stats["force"][k], np.ndarray):
                    stats["force"][k] = stats["force"][k].tolist()
        for int_key in ("num_trajectories", "num_transitions"):
            if int_key in stats and isinstance(stats[int_key], np.ndarray):
                stats[int_key] = stats[int_key].item()
    with open(out_path, "w") as f:
        json.dump(dataset_statistics, f, indent=2)
    print(f"Saved dataset statistics to {out_path}")


def tree_map(fn: Callable, tree: dict) -> dict:
    """Maps a function over a nested dictionary."""
    return {k: tree_map(fn, v) if isinstance(v, dict) else fn(v) for k, v in tree.items()}


def tree_map_with_key(fn: Callable, tree: dict, keys: Sequence = ()) -> dict:
    """Maps a function over a nested dictionary."""
    return {
        k: tree_map_with_key(fn, v, (*keys, k)) if isinstance(v, dict) else fn((*keys, k), v) for k, v in tree.items()
    }


@dataclass
class PaddedCollatorForLanguageModeling:
    model_max_length: int
    pad_token_id: int
    default_image_resolution: Tuple[int, int, int]
    padding_side: str = "right"
    pixel_values_dtype: torch.dtype = torch.float32

    def __post_init__(self) -> None:
        self.dummy_pixel_values = torch.zeros(self.default_image_resolution, dtype=self.pixel_values_dtype)

    def __call__(self, instances: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        pixel_values = [instance["pixel_values"] for instance in instances]

        # For now, we only support Tokenizers with `padding_side = "right"` during Training (but plan to extend!)
        #   => Handle padding via RNN Utils => `pad_sequence`
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.pad_token_id)
        labels = pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)

        # Truncate (if necessary)
        input_ids, labels = input_ids[:, : self.model_max_length], labels[:, : self.model_max_length]

        # Get `attention_mask` by checking for `pad_token_id`
        attention_mask = input_ids.ne(self.pad_token_id)

        # === Handle "unimodal" (language-only) vs. "multimodal" ===

        # Some examples are "language-only" --> build a Tensor of `multimodal_indices` that we can slice into easily
        multimodal_indices = torch.tensor(
            [idx for idx in range(len(pixel_values)) if pixel_values[idx] is not None], dtype=torch.long
        )

        # Stack all `pixel_values` --> depending on type (torch.Tensor, or Dict[str, torch.Tensor]) & presence of None
        if len(multimodal_indices) == 0:
            pixel_values = torch.stack([self.dummy_pixel_values for _ in range(len(input_ids))])
        elif isinstance(pv_example := pixel_values[multimodal_indices[0]], torch.Tensor):
            pixel_values = torch.stack(
                [
                    pixel_values[idx] if idx in multimodal_indices else self.dummy_pixel_values
                    for idx in range(len(input_ids))
                ]
            )
        elif isinstance(pv_example, dict):
            pixel_values = {
                k: torch.stack(
                    [
                        pixel_values[idx][k] if idx in multimodal_indices else self.dummy_pixel_values
                        for idx in range(len(input_ids))
                    ]
                )
                for k in pv_example
            }
        else:
            raise ValueError(f"Unsupported `pixel_values` type = {type(pixel_values)}")

        return dict(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            multimodal_indices=multimodal_indices,
        )


@dataclass
class PaddedCollatorForActionPrediction:
    model_max_length: int
    pad_token_id: int
    padding_side: str = "right"
    pixel_values_dtype: torch.dtype = torch.float32

    def __call__(self, instances: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        pixel_values = [instance["pixel_values"] for instance in instances]
        if "dataset_name" in instances[0]:
            dataset_names = [instance["dataset_name"] for instance in instances]
        else:
            dataset_names = None

        # For now, we only support Tokenizers with `padding_side = "right"` during training
        #   => Handle padding via RNN Utils => `pad_sequence`
        assert self.padding_side == "right", f"Invalid Tokenizer `{self.padding_side = }`"
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.pad_token_id)
        labels = pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)


        if self.padding_side == "left":
            def left_pad_sequence(sequences, padding_value):
                max_len = max(seq.size(0) for seq in sequences)
                padded = []
                for seq in sequences:
                    pad_len = max_len - seq.size(0)
                    pad = torch.full((pad_len,), padding_value, dtype=seq.dtype)
                    padded_seq = torch.cat([pad, seq], dim=0)
                    padded.append(padded_seq)
                return torch.stack(padded)

            input_ids = left_pad_sequence(input_ids, self.pad_token_id)
            labels = left_pad_sequence(labels, IGNORE_INDEX)
        else:
            input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.pad_token_id)
            labels = pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)


        # Truncate (if necessary)
        input_ids, labels = input_ids[:, : self.model_max_length], labels[:, : self.model_max_length]

        # Get `attention_mask` by checking for `pad_token_id`
        attention_mask = input_ids.ne(self.pad_token_id)

        # [Contract] For VLA Training =>> No "Unimodal" Data!
        assert all([pv is not None for pv in pixel_values]), "Invalid VLA Example with `pixel_values = None`!"

        # Stack all `pixel_values` --> depending on type is torch.Tensor or Dict[str, torch.Tensor]
        if isinstance(pixel_values[0], torch.Tensor):
            if "pixel_values_wrist" in instances[0]:
                pixel_values_wrist = [instance["pixel_values_wrist"] for instance in instances]
                pixel_values = torch.cat((torch.stack(pixel_values), torch.stack(pixel_values_wrist)), dim=1)
            else:
                pixel_values = torch.stack(pixel_values)
        else:
            raise ValueError(f"Unsupported `pixel_values` type = {type(pixel_values)}")

        # Stack all actions
        actions = [torch.from_numpy(np.copy(instance["actions"])) for instance in instances]
        actions = torch.stack(actions)

        # Stack proprio
        if "proprio" in instances[0]:
            proprio = [instance["proprio"] for instance in instances]
            proprio = torch.Tensor(np.squeeze(np.stack(proprio)))
        else:
            proprio = None

        # Stack wrist_depth (from RLBench depth adapter)
        if "wrist_depth" in instances[0]:
            wrist_depth = torch.stack([instance["wrist_depth"] for instance in instances])
        else:
            wrist_depth = None

        # Stack force_history (from RLBench force adapter)
        if "force_history" in instances[0]:
            force_history = torch.stack([instance["force_history"] for instance in instances])
        else:
            force_history = None

        # Stack phase_id (from RLBench phase annotations / dagger stage metadata)
        if any("phase_id" in instance for instance in instances):
            phase_id = torch.tensor([instance.get("phase_id", 0) for instance in instances], dtype=torch.long)
        else:
            phase_id = None

        output = dict(
            pixel_values=pixel_values,
            proprio=proprio,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            actions=actions,
        )
        if wrist_depth is not None:
            output["wrist_depth"] = wrist_depth
        if force_history is not None:
            output["force_history"] = force_history
        if phase_id is not None:
            output["phase_id"] = phase_id
        if dataset_names is not None:
            output["dataset_names"] = dataset_names
        return output
