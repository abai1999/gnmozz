"""
mixed_rlbench_dataset.py

Mix expert RLBench samples with planner-state DAgger samples using a fixed ratio.
"""

from torch.utils.data import Dataset


class MixedRLBenchDataset(Dataset):
    """Map-style wrapper that interleaves expert and DAgger samples by ratio."""

    def __init__(self, expert_dataset, dagger_dataset, expert_repeat: int = 2, dagger_repeat: int = 1):
        assert expert_repeat > 0 and dagger_repeat > 0
        self.expert_dataset = expert_dataset
        self.dagger_dataset = dagger_dataset
        self.expert_repeat = expert_repeat
        self.dagger_repeat = dagger_repeat
        self._expert_span = len(expert_dataset) * expert_repeat
        self._dagger_span = len(dagger_dataset) * dagger_repeat
        self._len = self._expert_span + self._dagger_span
        # Reuse the expert dataset normalization statistics so checkpoint saving and
        # downstream evaluation stay compatible with the original planner pipeline.
        self.dataset_statistics = getattr(expert_dataset, 'dataset_statistics', {})

    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        if idx < self._expert_span:
            sample = self.expert_dataset[idx % len(self.expert_dataset)]
            if "phase_id" not in sample:
                sample = dict(sample)
                sample["phase_id"] = 0
            return sample
        dagger_idx = idx - self._expert_span
        return self.dagger_dataset[dagger_idx % len(self.dagger_dataset)]
