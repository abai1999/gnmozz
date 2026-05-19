#!/usr/bin/env python
"""Audit phase1 grasp semantics for alignment TC student vNext.

Compares:
  - teacher target delta
  - student predicted target delta
  - student residual action

Axis-by-axis on phase1/grasp rows from the unified vNext dataset.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from prismatic.models.alignment_tc_student_vnext import AlignmentTCStudentVNext
from prismatic.vla.datasets.alignment_tc_student_vnext_dataset import AlignmentTCStudentVNextDataset


def _to6_from4(arr4: np.ndarray) -> np.ndarray:
    out = np.zeros((arr4.shape[0], 6), dtype=np.float32)
    out[:, :3] = arr4[:, :3]
    out[:, 5] = arr4[:, 3]
    return out


def _sign_rate(a: np.ndarray, b: np.ndarray, eps: float) -> float:
    active = (np.abs(a) > eps) | (np.abs(b) > eps)
    if not np.any(active):
        return float("nan")
    return float(np.mean(np.sign(a[active]) == np.sign(b[active])))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--output_json", required=True)
    ap.add_argument("--phase_id", type=int, default=0, help="0=grasp_commit, 1=insert_commit")
    ap.add_argument("--max_rows", type=int, default=2048)
    args = ap.parse_args()

    dataset = AlignmentTCStudentVNextDataset(args.dataset)
    data = np.load(args.dataset, allow_pickle=False)
    phase_id = data["phase_id"].astype(np.int64)
    verified_positive = data["verified_positive"].astype(np.float32)
    stage_bucket_id = data["stage_bucket_id"].astype(np.int64)

    phase_mask = phase_id == int(args.phase_id)
    idx = np.flatnonzero(phase_mask)
    if idx.size == 0:
        raise RuntimeError(f"No rows found for phase_id={args.phase_id}")
    if args.max_rows > 0 and idx.size > args.max_rows:
        idx = idx[: args.max_rows]

    ckpt = torch.load(args.ckpt, map_location="cpu")
    state_dict = ckpt.get("model_state_dict", ckpt)
    model = AlignmentTCStudentVNext(
        horizon=int(ckpt.get("horizon", 8)),
        max_pos_step=float(ckpt.get("max_pos_step", 0.0015)),
        max_yaw_step=float(ckpt.get("max_yaw_step", 0.0060)),
        y_bridge_max_step=float(ckpt.get("y_bridge_max_step", 0.001275)),
        use_front_rgb=bool(ckpt.get("use_front_rgb", False)),
        use_wrist_rgb=bool(ckpt.get("use_wrist_rgb", True)),
        use_wrist_depth=bool(ckpt.get("use_wrist_depth", True)),
        use_force=bool(ckpt.get("use_force", True)),
        use_planner_action=bool(ckpt.get("use_planner_action", True)),
    )
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    teacher_target = []
    teacher_action = []
    student_pred_target = []
    student_residual = []
    weights = []
    bucket_ids = []

    with torch.no_grad():
        for i in idx.tolist():
            batch = dataset[i]
            out = model(
                wrist_depth=batch["wrist_depth"].unsqueeze(0),
                force_history=batch["force_history"].unsqueeze(0),
                proprio=batch["proprio"].unsqueeze(0),
                planner_action_local=batch["planner_action_local"].unsqueeze(0),
                gripper_context=batch["gripper_context"].unsqueeze(0),
                front_rgb=batch["front_rgb"].unsqueeze(0),
                wrist_rgb=batch["wrist_rgb"].unsqueeze(0),
                phase_id=batch["phase_id"].unsqueeze(0),
                stage_bucket_id=batch["stage_bucket_id"].unsqueeze(0),
            )
            teacher_target.append(batch["teacher_target_delta_local_6d"].numpy())
            teacher_action.append(batch["teacher_residual_action_4d"].numpy())
            student_pred_target.append(out["pred_target_delta_local_6d"][0].cpu().numpy())
            student_residual.append(out["first_residual_6d"][0].cpu().numpy())
            weights.append(float(batch["sample_weight"].item()))
            bucket_ids.append(int(batch["stage_bucket_id"].item()))

    teacher_target = np.stack(teacher_target, axis=0).astype(np.float32)
    teacher_action_4d = np.stack(teacher_action, axis=0).astype(np.float32)
    teacher_action_6d = _to6_from4(teacher_action_4d)
    student_pred_target = np.stack(student_pred_target, axis=0).astype(np.float32)
    student_residual = np.stack(student_residual, axis=0).astype(np.float32)
    weights = np.asarray(weights, dtype=np.float32)
    bucket_ids = np.asarray(bucket_ids, dtype=np.int64)

    axis_spec = {
        "x": (0, 1e-5),
        "y": (1, 1e-5),
        "z": (2, 1e-5),
        "yaw": (5, 1e-4),
    }
    axis_report = {}
    for axis, (j, eps) in axis_spec.items():
        t = teacher_target[:, j]
        p = student_pred_target[:, j]
        r = student_residual[:, j]
        a = teacher_action_6d[:, j]
        axis_report[axis] = {
            "teacher_target_mean": float(np.mean(t)),
            "teacher_target_abs_mean": float(np.mean(np.abs(t))),
            "student_pred_target_mean": float(np.mean(p)),
            "student_pred_target_abs_mean": float(np.mean(np.abs(p))),
            "student_residual_mean": float(np.mean(r)),
            "student_residual_abs_mean": float(np.mean(np.abs(r))),
            "teacher_action_mean": float(np.mean(a)),
            "teacher_action_abs_mean": float(np.mean(np.abs(a))),
            "sign_rate_teacher_target_vs_student_pred": _sign_rate(t, p, eps),
            "sign_rate_teacher_target_vs_student_residual": _sign_rate(t, r, eps),
            "sign_rate_teacher_action_vs_student_residual": _sign_rate(a, r, eps),
        }

    bucket_hist = {}
    for b in np.unique(bucket_ids):
        bucket_hist[str(int(b))] = int(np.sum(bucket_ids == b))

    example_rows = []
    for k in range(min(12, teacher_target.shape[0])):
        example_rows.append(
            {
                "row_index": int(idx[k]),
                "stage_bucket_id": int(bucket_ids[k]),
                "verified_positive": float(verified_positive[idx[k]]),
                "sample_weight": float(weights[k]),
                "teacher_target_delta_6d": teacher_target[k].tolist(),
                "student_pred_target_delta_6d": student_pred_target[k].tolist(),
                "teacher_residual_action_6d": teacher_action_6d[k].tolist(),
                "student_residual_6d": student_residual[k].tolist(),
            }
        )

    out = {
        "dataset": str(Path(args.dataset).resolve()),
        "ckpt": str(Path(args.ckpt).resolve()),
        "phase_id": int(args.phase_id),
        "phase_name": "grasp_commit" if int(args.phase_id) == 0 else "insert_commit",
        "num_rows": int(teacher_target.shape[0]),
        "verified_positive_rate": float(np.mean(verified_positive[idx] > 0.5)),
        "sample_weight_mean": float(np.mean(weights)),
        "stage_bucket_id_hist": bucket_hist,
        "axis_report": axis_report,
        "example_rows": example_rows,
    }

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
