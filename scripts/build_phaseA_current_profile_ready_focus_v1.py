#!/usr/bin/env python3
"""
Post-process a Phase-A main dataset into a current-profile Ready-First focused
variant.

This keeps the existing runtime/far-negative/yaw-needed coverage intact, but
adds explicit current-profile episode flags and reweights rows so that
teacher-ready / ready-support / close-neighborhood windows are not drowned by
the broad runtime distribution.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _safe_candidates(rows, key):
    return [int(r["episode_index"]) for r in rows if int(r.get(key, 0)) > 0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_npz", required=True)
    ap.add_argument("--current_profile_positive_json", required=True)
    ap.add_argument("--current_profile_recollection_json", default="")
    ap.add_argument("--output_npz", required=True)
    ap.add_argument("--meta_json", required=True)
    ap.add_argument("--val_episode_csv_out", required=True)
    ap.add_argument("--teacher_ready_boost", type=float, default=1.8)
    ap.add_argument("--ready_support_boost", type=float, default=1.6)
    ap.add_argument("--current_profile_focus_boost", type=float, default=1.2)
    ap.add_argument("--close_near_boost", type=float, default=1.25)
    ap.add_argument("--val_teacher_ready_eps", type=int, default=2)
    ap.add_argument("--val_close_near_eps", type=int, default=1)
    ap.add_argument(
        "--positive_val_episode_csv",
        default="",
        help="Optional explicit positive validation episodes. Overrides automatic ready-positive val selection.",
    )
    ap.add_argument(
        "--hard_negative_episode_csv",
        default="",
        help="Current-profile hard-negative episodes; non-runtime rows are dropped and ready labels are forced negative.",
    )
    ap.add_argument("--hard_negative_boost", type=float, default=1.8)
    args = ap.parse_args()

    arr = np.load(args.input_npz, allow_pickle=False)
    data = {k: np.asarray(arr[k]) for k in arr.files}

    with open(args.current_profile_positive_json) as f:
        positive = json.load(f)
    recollection = {}
    if args.current_profile_recollection_json:
        with open(args.current_profile_recollection_json) as f:
            recollection = json.load(f)

    episode_index_full = np.asarray(data["episode_index"], dtype=np.int64)
    source_name_full = np.asarray(data.get("source_name", np.full((episode_index_full.shape[0],), "unknown", dtype="U32"))).astype(str)
    hard_negative_eps = {int(x) for x in args.hard_negative_episode_csv.split(",") if str(x).strip()}
    # For hard-negative episodes, keep the runtime-like rows and discard
    # teacher-assisted/oracle rows that may label the same episode as ready
    # under a different collection profile.
    hard_negative_ep_mask_full = np.isin(episode_index_full, sorted(hard_negative_eps))
    keep_mask = ~(hard_negative_ep_mask_full & (source_name_full != "runtime_like"))
    if not np.all(keep_mask):
        data = {
            k: (np.asarray(v)[keep_mask] if np.asarray(v).ndim >= 1 and np.asarray(v).shape[0] == keep_mask.shape[0] else v)
            for k, v in data.items()
        }

    episode_index = np.asarray(data["episode_index"], dtype=np.int64)
    source_name = np.asarray(data.get("source_name", np.full((episode_index.shape[0],), "unknown", dtype="U32"))).astype(str)
    teacher_ready = np.asarray(data["teacher_truth_handoff_ready"], dtype=np.float32) > 0.5
    ready_support = np.asarray(data["ready_support"], dtype=np.float32) > 0.5
    teacher_band = np.asarray(data["teacher_band_label"], dtype=np.int64)
    hard_negative_ep_mask = np.isin(episode_index, sorted(hard_negative_eps))
    # Hard-negative rows are current-profile negatives by construction.
    teacher_ready = teacher_ready & ~hard_negative_ep_mask
    ready_support = ready_support & ~hard_negative_ep_mask
    teacher_band = np.where(hard_negative_ep_mask, 0, teacher_band).astype(np.int64)

    positive_selected_eps = set(int(x) for x in positive.get("selected_episode_indices", []))
    recollection_selected_eps = set(int(x) for x in recollection.get("selected_episode_indices", []))
    selected_eps = set(positive_selected_eps)
    selected_eps.update(recollection_selected_eps)
    teacher_ready_candidates = set(_safe_candidates(positive.get("teacher_ready_candidates", []), "teacher_ready_rows"))
    close_near_candidates = set(_safe_candidates(positive.get("close_near_candidates", []), "close_near_rows"))

    dataset_teacher_ready_eps = sorted(int(x) for x in np.unique(episode_index[teacher_ready]).tolist())
    dataset_band_eps = sorted(int(x) for x in np.unique(episode_index[teacher_band >= 1]).tolist())

    current_profile_ready_eps = [
        ep for ep in dataset_teacher_ready_eps if ep in positive_selected_eps or ep in teacher_ready_candidates
    ]
    if not current_profile_ready_eps:
        current_profile_ready_eps = [
            ep for ep in dataset_teacher_ready_eps if ep in selected_eps or ep in teacher_ready_candidates
        ]
    if not current_profile_ready_eps:
        current_profile_ready_eps = dataset_teacher_ready_eps

    current_profile_close_near_eps = [
        ep for ep in dataset_band_eps if ep in positive_selected_eps or ep in close_near_candidates
    ]
    if not current_profile_close_near_eps:
        current_profile_close_near_eps = [ep for ep in dataset_band_eps if ep in selected_eps or ep in close_near_candidates]
    if not current_profile_close_near_eps:
        current_profile_close_near_eps = dataset_band_eps

    # Fixed validation episodes: keep a couple of current-profile ready-positive
    # episodes plus one close-neighborhood episode if available, but do not
    # consume every positive episode and starve training.
    explicit_positive_val_eps = [int(x) for x in args.positive_val_episode_csv.split(",") if str(x).strip()]
    val_eps: list[int] = []
    if explicit_positive_val_eps:
        for ep in explicit_positive_val_eps:
            if ep in dataset_teacher_ready_eps and ep not in val_eps:
                val_eps.append(ep)
    else:
        for ep in current_profile_ready_eps[: max(0, int(args.val_teacher_ready_eps))]:
            if ep not in val_eps:
                val_eps.append(ep)
    for ep in current_profile_close_near_eps:
        if len([x for x in val_eps if x in current_profile_close_near_eps]) >= int(args.val_close_near_eps):
            break
        if ep not in val_eps:
            val_eps.append(ep)
    for ep in sorted(hard_negative_eps):
        if ep in np.unique(episode_index).tolist() and ep not in val_eps:
            val_eps.append(ep)

    train_teacher_ready_eps = [ep for ep in dataset_teacher_ready_eps if ep not in val_eps]

    current_profile_focus = np.isin(episode_index, sorted(selected_eps)).astype(np.float32)
    current_profile_ready_episode = np.isin(episode_index, current_profile_ready_eps).astype(np.float32)
    current_profile_close_near_episode = np.isin(episode_index, current_profile_close_near_eps).astype(np.float32)
    current_profile_val_episode = np.isin(episode_index, val_eps).astype(np.float32)

    out = dict(data)
    out["teacher_truth_handoff_ready"] = teacher_ready.astype(np.float32)
    out["ready_support"] = ready_support.astype(np.float32)
    out["teacher_band_label"] = teacher_band.astype(np.int64)
    out["current_profile_focus_v1"] = current_profile_focus
    out["current_profile_ready_episode_v1"] = current_profile_ready_episode
    out["current_profile_close_near_episode_v1"] = current_profile_close_near_episode
    out["current_profile_val_episode_v1"] = current_profile_val_episode
    out["current_profile_hard_negative_v1"] = hard_negative_ep_mask.astype(np.float32)

    sw = np.asarray(out.get("sample_weight", np.ones_like(current_profile_focus)), dtype=np.float32).copy()
    sw *= np.where(current_profile_focus > 0.5, float(args.current_profile_focus_boost), 1.0).astype(np.float32)
    sw *= np.where(current_profile_close_near_episode > 0.5, float(args.close_near_boost), 1.0).astype(np.float32)
    sw *= np.where(teacher_ready, float(args.teacher_ready_boost), 1.0).astype(np.float32)
    sw *= np.where(ready_support, float(args.ready_support_boost), 1.0).astype(np.float32)
    sw *= np.where(hard_negative_ep_mask, float(args.hard_negative_boost), 1.0).astype(np.float32)
    out["sample_weight"] = sw.astype(np.float32)

    output_npz = Path(args.output_npz)
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_npz, **out)

    meta = {
        "input_npz": str(args.input_npz),
        "output_npz": str(output_npz),
        "dataset_teacher_ready_eps": dataset_teacher_ready_eps,
        "dataset_band_eps": dataset_band_eps,
        "current_profile_selected_eps": sorted(selected_eps),
        "positive_selected_eps": sorted(positive_selected_eps),
        "recollection_selected_eps": sorted(recollection_selected_eps),
        "current_profile_ready_eps": current_profile_ready_eps,
        "current_profile_close_near_eps": current_profile_close_near_eps,
        "hard_negative_eps": sorted(hard_negative_eps),
        "dropped_hard_negative_non_runtime_rows": int(np.sum(~keep_mask)),
        "recommended_val_episode_indices": val_eps,
        "recommended_val_episode_csv": ",".join(str(x) for x in val_eps),
        "recommended_train_teacher_ready_eps": train_teacher_ready_eps,
        "counts": {
            "rows": int(episode_index.shape[0]),
            "source_rows_after_drop": {
                str(src): int(np.sum(source_name == src)) for src in np.unique(source_name).tolist()
            },
            "teacher_ready_rows": int(np.sum(teacher_ready)),
            "teacher_ready_eps": int(len(dataset_teacher_ready_eps)),
            "ready_support_rows": int(np.sum(ready_support)),
            "ready_support_eps": int(np.unique(episode_index[ready_support]).size) if np.any(ready_support) else 0,
            "val_teacher_ready_rows": int(np.sum(teacher_ready & (current_profile_val_episode > 0.5))),
            "val_ready_support_rows": int(np.sum(ready_support & (current_profile_val_episode > 0.5))),
            "train_teacher_ready_rows": int(np.sum(teacher_ready & (current_profile_val_episode <= 0.5))),
            "train_ready_support_rows": int(np.sum(ready_support & (current_profile_val_episode <= 0.5))),
            "hard_negative_rows": int(np.sum(hard_negative_ep_mask)),
            "hard_negative_val_rows": int(np.sum(hard_negative_ep_mask & (current_profile_val_episode > 0.5))),
        },
        "weights": {
            "teacher_ready_boost": float(args.teacher_ready_boost),
            "ready_support_boost": float(args.ready_support_boost),
            "current_profile_focus_boost": float(args.current_profile_focus_boost),
            "close_near_boost": float(args.close_near_boost),
            "hard_negative_boost": float(args.hard_negative_boost),
        },
    }
    meta_path = Path(args.meta_json)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, indent=2))
    Path(args.val_episode_csv_out).write_text(meta["recommended_val_episode_csv"] + "\n")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
