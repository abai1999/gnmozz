#!/usr/bin/env python3
"""Offline task audit for insert_onto_square_peg."""

from __future__ import annotations

import argparse
import json
import pickle
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_pickle(path: Path):
    return pickle.load(open(path, "rb"))


def _episode_index(episode_dir: Path) -> int:
    name = episode_dir.name
    if name.startswith("episode"):
        try:
            return int(name.replace("episode", ""))
        except Exception:
            return -1
    return -1


def _find_episode_dirs(root: Path) -> list[Path]:
    episodes_root = root / "train" / "episodes"
    if not episodes_root.exists():
        return []
    return sorted(
        [p for p in episodes_root.iterdir() if p.is_dir() and p.name.startswith("episode")],
        key=_episode_index,
    )


def _phase_first_indices(phase_ids: np.ndarray) -> dict[str, int]:
    phase_to_first: dict[int, int] = {}
    for idx, phase_id in enumerate(np.asarray(phase_ids).reshape(-1).tolist()):
        phase_to_first.setdefault(int(phase_id), int(idx))
    return phase_to_first


def _gripper_transition_indices(gripper_open: np.ndarray) -> dict[str, int | None]:
    arr = np.asarray(gripper_open, dtype=np.float32).reshape(-1)
    close_idx = next((int(i) for i, value in enumerate(arr) if value < 0.5), None)
    reopen_idx = None
    if close_idx is not None:
        reopen_idx = next((int(i) for i in range(close_idx + 1, len(arr)) if arr[i] >= 0.5), None)
    return {"close_idx": close_idx, "reopen_idx": reopen_idx}


def _force_summary(forces: np.ndarray) -> dict:
    arr = np.asarray(forces, dtype=np.float32)
    if arr.ndim != 2:
        arr = arr.reshape(len(arr), -1)
    norms = np.linalg.norm(arr[:, :3], axis=1)
    return {
        "mean_force_norm": float(np.mean(norms)) if norms.size else 0.0,
        "std_force_norm": float(np.std(norms)) if norms.size else 0.0,
        "p95_force_norm": float(np.percentile(norms, 95)) if norms.size else 0.0,
        "max_force_norm": float(np.max(norms)) if norms.size else 0.0,
        "mean_force_xyz": [float(x) for x in np.mean(arr[:, :3], axis=0)] if arr.size else [0.0, 0.0, 0.0],
        "mean_torque_xyz": [float(x) for x in np.mean(arr[:, 3:6], axis=0)] if arr.shape[1] >= 6 else [0.0, 0.0, 0.0],
    }


def _phase_summary(phase_annotation: dict) -> dict:
    return {
        "phase_counts": dict(phase_annotation.get("phase_counts", {})),
        "grasp_confirmed": bool(phase_annotation.get("grasp_confirmed", False)),
        "contact_detected": bool(phase_annotation.get("contact_detected", False)),
        "contact_events": phase_annotation.get("contact_events", []),
        "phase_trigger_config": phase_annotation.get("phase_trigger_config", {}),
        "refine_force_feedback": phase_annotation.get("refine_force_feedback", {}),
    }


def _resize_keep_aspect(img: Image.Image, target_h: int) -> Image.Image:
    if img.height == target_h:
        return img
    scale = target_h / float(max(img.height, 1))
    new_w = max(1, int(round(img.width * scale)))
    return img.resize((new_w, target_h), Image.BILINEAR)


def _make_contact_sheet(episode_dir: Path, frame_indices: list[int], labels: list[str], output_path: Path) -> None:
    images = []
    for idx, label in zip(frame_indices, labels):
        frame_path = episode_dir / "front_rgb" / f"{idx}.png"
        if not frame_path.exists():
            continue
        img = Image.open(frame_path).convert("RGB")
        img = _resize_keep_aspect(img, 180)
        canvas = Image.new("RGB", (img.width, img.height + 32), color=(245, 245, 245))
        canvas.paste(img, (0, 0))
        draw = ImageDraw.Draw(canvas)
        text = f"{label} | {idx}"
        draw.rectangle([0, img.height, canvas.width, canvas.height], fill=(245, 245, 245))
        draw.text((6, img.height + 6), text, fill=(20, 20, 20))
        images.append(canvas)

    if not images:
        return
    sheet_w = sum(img.width for img in images)
    sheet_h = max(img.height for img in images)
    sheet = Image.new("RGB", (sheet_w, sheet_h), color=(255, 255, 255))
    x = 0
    for img in images:
        sheet.paste(img, (x, 0))
        x += img.width
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("."))
    ap.add_argument("--task_root", type=Path, default=Path("data/rlbench_data/insert_onto_square_peg"))
    ap.add_argument(
        "--output_root",
        type=Path,
        default=Path("runtime_artifacts/coarse2contact/task_audit/insert_onto_square_peg"),
    )
    ap.add_argument("--contact_sheet_limit", type=int, default=20)
    args = ap.parse_args()

    root = args.root.resolve()
    task_root = (root / args.task_root).resolve()
    output_root = (root / args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    sheet_root = output_root / "contact_sheets"
    sheet_root.mkdir(parents=True, exist_ok=True)

    task_summary = {
        "task_name": "insert_onto_square_peg",
        "task_root": str(task_root),
        "descriptions": [],
        "num_episodes": 0,
        "phase_totals": Counter(),
        "gripper_transition_hist": Counter(),
        "force_norms": [],
        "episodes": [],
    }

    episode_dirs = _find_episode_dirs(task_root)
    for ep_dir in episode_dirs:
        ep_idx = _episode_index(ep_dir)
        phase_path = ep_dir / "phase_annotation.json"
        phase_ids_path = ep_dir / "phase_ids.npy"
        inputs_path = ep_dir / "model_inputs.npz"
        desc_path = ep_dir / "variation_descriptions.pkl"
        if not (phase_path.exists() and phase_ids_path.exists() and inputs_path.exists() and desc_path.exists()):
            continue

        phase_annotation = _load_json(phase_path)
        phase_ids = np.load(phase_ids_path)
        model_inputs = np.load(inputs_path, allow_pickle=True)
        descriptions = _load_pickle(desc_path)
        if isinstance(descriptions, (list, tuple)):
            task_summary["descriptions"].extend(str(x) for x in descriptions)

        force_stats = _force_summary(model_inputs["gripper_touch_forces"])
        gripper_transitions = _gripper_transition_indices(model_inputs["gripper_open"])
        phase_first = _phase_first_indices(phase_ids)
        phase_counts = phase_annotation.get("phase_counts", {})

        for k, v in phase_counts.items():
            task_summary["phase_totals"][str(k)] += int(v)
        task_summary["force_norms"].append(force_stats["mean_force_norm"])
        task_summary["gripper_transition_hist"]["close"] += int(gripper_transitions["close_idx"] is not None)
        task_summary["gripper_transition_hist"]["reopen"] += int(gripper_transitions["reopen_idx"] is not None)

        contact_events = phase_annotation.get("contact_events", [])
        contact_idx = int(contact_events[0].get("contact_idx", -1)) if contact_events else -1
        pre_contact_idx = int(contact_events[0].get("pre_contact_idx", -1)) if contact_events else -1
        refine_phase_idx = int(phase_first.get(3, -1))
        grasp_phase_idx = int(phase_first.get(1, -1))
        transfer_phase_idx = int(phase_first.get(2, -1))
        reach_phase_idx = int(phase_first.get(0, -1))
        last_idx = int(phase_ids.shape[0] - 1)
        frame_indices = [
            reach_phase_idx,
            grasp_phase_idx,
            transfer_phase_idx,
            pre_contact_idx,
            contact_idx,
            refine_phase_idx,
            last_idx,
        ]
        labels = [
            "Reach",
            "Grasp",
            "Transfer",
            "Pre-contact",
            "Contact",
            "Refine",
            "Final",
        ]
        compact = []
        compact_labels = []
        seen = set()
        for idx, label in zip(frame_indices, labels):
            if idx < 0 or idx in seen:
                continue
            seen.add(idx)
            compact.append(idx)
            compact_labels.append(label)

        if len(list(sheet_root.glob("*.jpg"))) < args.contact_sheet_limit:
            _make_contact_sheet(ep_dir, compact, compact_labels, sheet_root / f"episode{ep_idx:03d}.jpg")

        task_summary["episodes"].append(
            {
                "episode_index": ep_idx,
                "num_steps": int(phase_annotation.get("num_steps", len(phase_ids))),
                "descriptions": list(descriptions) if isinstance(descriptions, (list, tuple)) else [str(descriptions)],
                "phase_counts": dict(phase_counts),
                "gripper_transition": gripper_transitions,
                "contact_events": contact_events,
                "force_stats": force_stats,
                "contact_sheet": str(sheet_root / f"episode{ep_idx:03d}.jpg"),
            }
        )

    task_summary["num_episodes"] = len(task_summary["episodes"])
    if task_summary["force_norms"]:
        arr = np.asarray(task_summary["force_norms"], dtype=np.float32)
        task_summary["force_norm_summary"] = {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "p95": float(np.percentile(arr, 95)),
            "max": float(np.max(arr)),
        }
    else:
        task_summary["force_norm_summary"] = {"mean": 0.0, "std": 0.0, "p95": 0.0, "max": 0.0}
    task_summary["phase_totals"] = dict(task_summary["phase_totals"])
    task_summary["gripper_transition_hist"] = dict(task_summary["gripper_transition_hist"])

    aggregate_path = output_root / "insert_onto_square_peg_audit_summary.json"
    aggregate_path.write_text(json.dumps(task_summary, indent=2, sort_keys=True), encoding="utf-8")
    print(aggregate_path)


if __name__ == "__main__":
    main()
