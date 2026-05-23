#!/usr/bin/env python3
"""Build a force/contact classifier dataset for Coarse2Contact v2."""

from __future__ import annotations

import argparse
import json
import pickle
from collections import Counter
from pathlib import Path

import numpy as np


def _episode_dirs(root: Path) -> list[Path]:
    episodes_root = root / "train" / "episodes"
    if not episodes_root.exists():
        return []

    def _episode_idx(p: Path) -> int:
        try:
            return int(p.name.replace("episode", ""))
        except Exception:
            return -1

    return sorted([p for p in episodes_root.iterdir() if p.is_dir() and p.name.startswith("episode")], key=_episode_idx)


def _load_pickle(path: Path):
    with open(path, "rb") as handle:
        return pickle.load(handle)


def _phase_name(phase_annotation: dict, phase_id: int) -> str:
    mapping = phase_annotation.get("phase_id_to_name", {})
    return str(mapping.get(str(int(phase_id)), mapping.get(int(phase_id), f"phase_{phase_id}")))


def _force_summary(forces: np.ndarray) -> dict[str, float]:
    arr = np.asarray(forces, dtype=np.float32)
    if arr.ndim != 2:
        arr = arr.reshape(len(arr), -1)
    norms = np.linalg.norm(arr[:, :3], axis=1)
    torques = np.linalg.norm(arr[:, 3:6], axis=1) if arr.shape[1] >= 6 else np.zeros_like(norms)
    return {
        "mean_force_norm": float(np.mean(norms)) if norms.size else 0.0,
        "p95_force_norm": float(np.percentile(norms, 95)) if norms.size else 0.0,
        "p99_force_norm": float(np.percentile(norms, 99)) if norms.size else 0.0,
        "mean_torque_norm": float(np.mean(torques)) if torques.size else 0.0,
        "p95_torque_norm": float(np.percentile(torques, 95)) if torques.size else 0.0,
        "p99_torque_norm": float(np.percentile(torques, 99)) if torques.size else 0.0,
    }


def _moving_average(arr: np.ndarray, window: int = 5) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[:, None]
    out = np.zeros_like(arr)
    for i in range(len(arr)):
        lo = max(0, i - window + 1)
        out[i] = np.mean(arr[lo : i + 1], axis=0)
    return out


def _label_sample(
    *,
    t: int,
    phase_id: int,
    phase_name: str,
    force_norm: np.ndarray,
    torque_norm: np.ndarray,
    gripper_open: np.ndarray,
    phase_annotation: dict,
    free_force_q95: float,
    jam_force_q99: float,
    torque_q95: float,
    invalid_action_flag: np.ndarray | None = None,
) -> dict[str, int]:
    contact_events = phase_annotation.get("contact_events", [])
    contact_idx_set = set()
    for event in contact_events:
        contact_idx = int(event.get("contact_idx", -1))
        if contact_idx >= 0:
            contact_idx_set.update(range(max(0, contact_idx - 1), contact_idx + 2))
    contact = int(force_norm[t] >= free_force_q95 or t in contact_idx_set or phase_name in {"Grasp", "Refine"})
    jam = int(force_norm[t] >= jam_force_q99 or torque_norm[t] >= torque_q95 or (invalid_action_flag is not None and bool(invalid_action_flag[t])))
    grasp_confirmed = bool(phase_annotation.get("grasp_confirmed", False))
    misgrasp = int(phase_name == "Grasp" and not grasp_confirmed)
    slip = int(
        phase_name in {"Transfer", "Refine"}
        and gripper_open[t] < 0.5
        and t > 0
        and force_norm[t - 1] >= free_force_q95
        and force_norm[t] < 0.7 * free_force_q95
    )
    recovery_needed = int(bool(jam or misgrasp or slip or (invalid_action_flag is not None and bool(np.any(invalid_action_flag[max(0, t - 2) : t + 1])))))
    return {
        "contact": contact,
        "jam": jam,
        "misgrasp": misgrasp,
        "slip": slip,
        "recovery_needed": recovery_needed,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task_root", type=Path, default=Path("data/rlbench_data/insert_onto_square_peg"))
    ap.add_argument("--output_root", type=Path, default=Path("runtime_artifacts/coarse2contact_v2/datasets"))
    ap.add_argument("--window_len", type=int, default=12)
    ap.add_argument("--sample_stride", type=int, default=2)
    args = ap.parse_args()

    root = args.task_root.resolve()
    out_root = args.output_root.resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    out_path = out_root / "force_contact_dataset.jsonl"

    samples = []
    summary = {
        "task_name": "insert_onto_square_peg",
        "task_root": str(root),
        "output_path": str(out_path),
        "label_counts": Counter(),
        "phase_counts": Counter(),
        "episodes": [],
    }

    for ep_dir in _episode_dirs(root):
        ep_idx = int(ep_dir.name.replace("episode", ""))
        phase_path = ep_dir / "phase_annotation.json"
        phase_ids_path = ep_dir / "phase_ids.npy"
        inputs_path = ep_dir / "model_inputs.npz"
        if not (phase_path.exists() and phase_ids_path.exists() and inputs_path.exists()):
            continue
        phase_annotation = json.loads(phase_path.read_text(encoding="utf-8"))
        phase_ids = np.load(phase_ids_path)
        model_inputs = np.load(inputs_path, allow_pickle=True)
        forces = np.asarray(model_inputs["gripper_touch_forces"], dtype=np.float32)
        gripper_open = np.asarray(model_inputs["gripper_open"], dtype=np.float32).reshape(-1)
        gripper_pose = np.asarray(model_inputs["gripper_pose"], dtype=np.float32)
        action_targets = np.asarray(model_inputs["action_targets"], dtype=np.float32)
        proprio = np.asarray(model_inputs["proprio"], dtype=np.float32)
        force_norm = np.linalg.norm(forces[:, :3], axis=1)
        torque_norm = np.linalg.norm(forces[:, 3:6], axis=1) if forces.shape[1] >= 6 else np.zeros_like(force_norm)

        phase0_mask = np.asarray(phase_ids).reshape(-1) == 0
        free_force_q95 = float(np.percentile(force_norm[phase0_mask], 95)) if np.any(phase0_mask) else float(np.percentile(force_norm, 95))
        jam_force_q99 = float(np.percentile(force_norm, 99)) if force_norm.size else 0.0
        torque_q95 = float(np.percentile(torque_norm, 95)) if torque_norm.size else 0.0
        invalid_action_flag = np.zeros_like(force_norm, dtype=bool)

        episode_labels = Counter()
        for t in range(0, len(phase_ids), max(int(args.sample_stride), 1)):
            phase_id = int(phase_ids[t])
            phase_name = _phase_name(phase_annotation, phase_id)
            summary["phase_counts"][phase_name] += 1
            window_start = max(0, t - int(args.window_len) + 1)
            window_end = t + 1
            labels = _label_sample(
                t=t,
                phase_id=phase_id,
                phase_name=phase_name,
                force_norm=force_norm,
                torque_norm=torque_norm,
                gripper_open=gripper_open,
                phase_annotation=phase_annotation,
                free_force_q95=free_force_q95,
                jam_force_q99=jam_force_q99,
                torque_q95=torque_q95,
                invalid_action_flag=invalid_action_flag,
            )
            sample = {
                "task_name": "insert_onto_square_peg",
                "episode_idx": ep_idx,
                "step_idx": int(t),
                "window_start": int(window_start),
                "window_end": int(window_end),
                "window_len": int(args.window_len),
                "phase_id": phase_id,
                "phase_name": phase_name,
                "skill_type": "force_contact_classifier",
                "stage_name": "RING_GRASP_CONTACT" if phase_name in {"Grasp"} else ("SLIDE_ON_SPOKE" if phase_name in {"Transfer", "Refine"} else "COARSE_TO_RING"),
                "npz_path": str(inputs_path),
                "gripper_pose": gripper_pose[t].tolist(),
                "proprio": proprio[t].tolist(),
                "action_target": action_targets[t].tolist(),
                "force_norm": float(force_norm[t]),
                "torque_norm": float(torque_norm[t]),
                "free_force_q95": free_force_q95,
                "jam_force_q99": jam_force_q99,
                "torque_q95": torque_q95,
                "label_contact": int(labels["contact"]),
                "label_jam": int(labels["jam"]),
                "label_misgrasp": int(labels["misgrasp"]),
                "label_slip": int(labels["slip"]),
                "label_recovery_needed": int(labels["recovery_needed"]),
                "label_source": "privileged_phase_and_force_heuristic",
                "uses_privileged_label": True,
                "uses_privileged_runtime": False,
                "sample_kind": "positive" if labels["recovery_needed"] else "negative",
                "language_targets": list(phase_annotation.get("language_targets", [])),
            }
            samples.append(sample)
            for key, value in labels.items():
                summary["label_counts"][key] += int(value)
                episode_labels[key] += int(value)

        summary["episodes"].append(
            {
                "episode_idx": ep_idx,
                "num_steps": int(len(phase_ids)),
                "force_summary": {
                    "mean_force_norm": float(np.mean(force_norm)) if force_norm.size else 0.0,
                    "p95_force_norm": float(np.percentile(force_norm, 95)) if force_norm.size else 0.0,
                    "p99_force_norm": float(np.percentile(force_norm, 99)) if force_norm.size else 0.0,
                    "mean_torque_norm": float(np.mean(torque_norm)) if torque_norm.size else 0.0,
                },
                "label_counts": dict(episode_labels),
            }
        )

    with open(out_path, "w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(sample, sort_keys=True) + "\n")

    summary["label_counts"] = dict(summary["label_counts"])
    summary["phase_counts"] = dict(summary["phase_counts"])
    summary["num_samples"] = len(samples)
    summary_path = out_root / "force_contact_dataset_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(out_path)
    print(summary_path)


if __name__ == "__main__":
    main()
