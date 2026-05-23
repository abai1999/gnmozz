"""Dataset helpers for Coarse2Contact v2 learned modules."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


def read_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def build_vocab(values: Iterable[str], *, add_unknown: bool = True) -> dict[str, int]:
    vocab: dict[str, int] = {}
    if add_unknown:
        vocab["<unk>"] = 0
    for value in values:
        key = str(value)
        if key not in vocab:
            vocab[key] = len(vocab)
    return vocab


def _lookup(vocab: Mapping[str, int], value: str) -> int:
    if value in vocab:
        return int(vocab[value])
    return int(vocab.get("<unk>", 0))


def _center_crop_box(width: int, height: int, crop_size: int) -> tuple[int, int, int, int]:
    crop = int(max(8, min(crop_size, min(width, height))))
    half = crop // 2
    cx = width // 2
    cy = height // 2
    x0 = max(0, cx - half)
    y0 = max(0, cy - half)
    x1 = min(width, x0 + crop)
    y1 = min(height, y0 + crop)
    if x1 - x0 < crop:
        x0 = max(0, x1 - crop)
    if y1 - y0 < crop:
        y0 = max(0, y1 - crop)
    return int(x0), int(y0), int(x1), int(y1)


def _load_rgbd(rgb_path: Path, depth_path: Path, *, crop_box: tuple[int, int, int, int] | None = None, resize_to: int | None = None) -> torch.Tensor:
    rgb_img = Image.open(rgb_path).convert("RGB")
    depth_img = Image.open(depth_path)
    if crop_box is not None:
        rgb_img = rgb_img.crop(crop_box)
        depth_img = depth_img.crop(crop_box)
    if resize_to is not None and resize_to > 0:
        rgb_img = rgb_img.resize((resize_to, resize_to), resample=Image.BILINEAR)
        depth_img = depth_img.resize((resize_to, resize_to), resample=Image.BILINEAR)
    rgb = np.asarray(rgb_img, dtype=np.float32) / 255.0
    depth = np.asarray(depth_img, dtype=np.float32)
    if depth.ndim == 3:
        depth = depth[..., 0]
    if float(depth.max()) > 1.5:
        depth = depth / 255.0
    depth = np.clip(depth, 0.0, 1.0)
    rgbd = np.concatenate([rgb, depth[..., None]], axis=-1)
    h, w = rgbd.shape[:2]
    xs = np.linspace(-1.0, 1.0, num=w, dtype=np.float32)
    ys = np.linspace(-1.0, 1.0, num=h, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(xs, ys)
    rgbd = np.concatenate([rgbd, grid_x[..., None], grid_y[..., None]], axis=-1)
    return torch.from_numpy(np.transpose(rgbd, (2, 0, 1))).float()


def _moving_average(arr: np.ndarray, window: int = 5) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[:, None]
    out = np.zeros_like(arr)
    for i in range(len(arr)):
        lo = max(0, i - window + 1)
        out[i] = np.mean(arr[lo : i + 1], axis=0)
    return out


class DepthLocalizerJsonlDataset(Dataset):
    def __init__(self, jsonl_path: Path, *, records: list[dict] | None = None) -> None:
        self.jsonl_path = Path(jsonl_path)
        self.records = records if records is not None else read_jsonl(self.jsonl_path)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        record = dict(self.records[idx])
        crop_box = record.get("roi_box")
        if crop_box is not None and len(crop_box) == 4:
            crop_box = tuple(int(x) for x in crop_box)
        else:
            crop_box = None
        resize_to = int(record.get("roi_resize_px", 128) or 128)
        rgbd = _load_rgbd(Path(record["rgb_path"]), Path(record["depth_path"]), crop_box=crop_box, resize_to=resize_to)
        record["image_rgbd"] = rgbd
        record["skill_type_id"] = record["skill_type"]
        record["stage_id"] = record["stage_name"]
        record["target_entity_id"] = record["target_entity"]
        record["reference_entity_id"] = record["reference_entity"]
        record["controlled_dofs_vec"] = list(record.get("controlled_dofs", []))
        return record

    def build_vocab(self) -> dict[str, dict[str, int]]:
        return {
            "skill_type": build_vocab(r["skill_type"] for r in self.records),
            "stage_name": build_vocab(r["stage_name"] for r in self.records),
            "entity": build_vocab(
                list(r["target_entity"] for r in self.records) + list(r["reference_entity"] for r in self.records)
            ),
            "controlled_dofs": build_vocab(
                dof for r in self.records for dof in r.get("controlled_dofs", [])
            ),
        }


class ForceContactJsonlDataset(Dataset):
    def __init__(self, jsonl_path: Path, *, records: list[dict] | None = None) -> None:
        self.jsonl_path = Path(jsonl_path)
        self.records = records if records is not None else read_jsonl(self.jsonl_path)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        record = dict(self.records[idx])
        npz = np.load(record["npz_path"], allow_pickle=True)
        start = int(record["window_start"])
        end = int(record["window_end"])
        forces = np.asarray(npz["gripper_touch_forces"], dtype=np.float32)[start:end]
        gripper_open = np.asarray(npz["gripper_open"], dtype=np.float32).reshape(-1)[start:end]
        gripper_pose = np.asarray(npz["gripper_pose"], dtype=np.float32)[start:end]
        action_targets = np.asarray(npz["action_targets"], dtype=np.float32)[start:end]
        if forces.ndim != 2:
            forces = forces.reshape(len(forces), -1)
        raw = forces[:, :6]
        filt = _moving_average(raw, window=5)
        delta = np.zeros_like(raw)
        if len(raw) > 1:
            delta[1:] = raw[1:] - raw[:-1]
        z0 = float(gripper_pose[0, 2]) if len(gripper_pose) else 0.0
        z_progress = gripper_pose[:, 2:3] - z0
        invalid = np.zeros((len(raw), 1), dtype=np.float32)
        action_hist = action_targets[:, :6]
        features = np.concatenate([raw, filt, delta, gripper_open[:, None], z_progress, invalid, action_hist], axis=-1)
        record["sequence_features"] = torch.from_numpy(features).float()
        record["skill_type_id"] = record["skill_type"]
        record["stage_id"] = record["stage_name"]
        return record

    def build_vocab(self) -> dict[str, dict[str, int]]:
        return {
            "skill_type": build_vocab(r["skill_type"] for r in self.records),
            "stage_name": build_vocab(r["stage_name"] for r in self.records),
        }
