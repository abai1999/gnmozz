"""
build_fire_donor_distill_dataset.py

Convert donor support-state dumps from random-state alignment rollouts into a
clean fire-only distillation dataset.

The key rule is to keep fire supervision anchor-local:
- positives are only the final 1-2 ALIGN+open rows before the donor fire anchor
- negatives come from clearly earlier rows or early-close episodes
- episode-level success signals are not copied onto every prior row
"""

import argparse
import json
from pathlib import Path

import numpy as np


def load_npz(path: Path) -> dict:
    data = np.load(path, allow_pickle=True)
    return {k: data[k] for k in data.files}


def save_npz(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **data)


def slice_rows(data: dict, indices: np.ndarray) -> dict:
    out = {}
    for key, value in data.items():
        arr = np.asarray(value)
        if arr.shape[:1] == (len(next(iter(data.values()))),):
            out[key] = arr[indices]
        else:
            out[key] = arr
    return out


def relabel_and_select(
    raw: dict,
    *,
    source_npz: str,
    positive_window: int,
    hard_negative_gap: int,
    max_negative_ratio: int,
    seed: int,
):
    episode_ids = np.asarray(raw["episode_index"], dtype=np.int64)
    rollout_step = np.asarray(raw["rollout_step"], dtype=np.int64)
    ready_old = np.asarray(raw.get("ready_to_close_target", np.zeros_like(episode_ids, dtype=np.float32)), dtype=np.float32)
    lift_old = np.asarray(raw.get("grasp_lift_proxy", np.zeros_like(episode_ids, dtype=np.float32)), dtype=np.float32)
    stable_old = np.asarray(raw.get("post_close_stability_proxy", np.zeros_like(episode_ids, dtype=np.float32)), dtype=np.float32)
    reopen_old = np.asarray(raw.get("reopen_after_trigger", np.zeros_like(episode_ids, dtype=np.float32)), dtype=np.float32)
    invalid_old = np.asarray(raw.get("invalid_after_trigger", np.zeros_like(episode_ids, dtype=np.float32)), dtype=np.float32)
    early_old = np.asarray(raw.get("planner_close_too_early", np.zeros_like(episode_ids, dtype=np.float32)), dtype=np.float32)

    out = {k: np.asarray(v).copy() for k, v in raw.items()}
    num_rows = int(episode_ids.shape[0])
    out["ready_to_close_target"] = np.zeros((num_rows,), dtype=np.float32)
    out["post_close_stability_proxy"] = np.zeros((num_rows,), dtype=np.float32)
    out["grasp_lift_proxy"] = np.zeros((num_rows,), dtype=np.float32)
    out["reopen_after_trigger"] = np.zeros((num_rows,), dtype=np.float32)
    out["invalid_after_trigger"] = np.zeros((num_rows,), dtype=np.float32)
    out["planner_close_too_early"] = np.zeros((num_rows,), dtype=np.float32)
    out["fire_label_kind"] = np.full((num_rows,), -1, dtype=np.int64)

    rng = np.random.default_rng(int(seed))
    keep_indices = []
    episode_summary = []
    total_positive = 0
    total_negative = 0

    for ep in np.unique(episode_ids):
        ep_mask = episode_ids == int(ep)
        ep_idx = np.where(ep_mask)[0]
        ep_steps = rollout_step[ep_idx]
        ep_ready = ready_old[ep_idx] > 0.5
        ep_lift = bool(np.max(lift_old[ep_idx]) > 0.5 or np.max(stable_old[ep_idx]) > 0.5 or np.any(ep_ready))
        anchor_step = int(np.max(ep_steps[ep_ready])) if np.any(ep_ready) else None

        if anchor_step is not None:
            pos_mask = (anchor_step - ep_steps >= 0) & (anchor_step - ep_steps <= int(positive_window))
            far_before_anchor = (anchor_step - ep_steps) >= int(hard_negative_gap)
        else:
            pos_mask = np.zeros_like(ep_steps, dtype=bool)
            far_before_anchor = np.zeros_like(ep_steps, dtype=bool)

        neg_flag_mask = (early_old[ep_idx] > 0.5) | (reopen_old[ep_idx] > 0.5) | (invalid_old[ep_idx] > 0.5)
        neg_mask = neg_flag_mask | far_before_anchor

        pos_idx = ep_idx[pos_mask]
        neg_idx = ep_idx[neg_mask & (~pos_mask)]

        max_neg = max(int(len(pos_idx) * int(max_negative_ratio)), int(4 if len(pos_idx) > 0 else 24))
        if len(neg_idx) > max_neg:
            flagged = ep_idx[neg_flag_mask & (~pos_mask)]
            selected = []
            if len(flagged) > 0:
                if len(flagged) > max_neg:
                    pick = rng.choice(flagged, size=max_neg, replace=False)
                    selected.extend(int(x) for x in pick.tolist())
                else:
                    selected.extend(int(x) for x in flagged.tolist())
            remain = [int(x) for x in neg_idx.tolist() if int(x) not in set(selected)]
            if len(selected) < max_neg and remain:
                pick = rng.choice(np.asarray(remain, dtype=np.int64), size=min(len(remain), max_neg - len(selected)), replace=False)
                selected.extend(int(x) for x in pick.tolist())
            neg_idx = np.asarray(sorted(set(selected)), dtype=np.int64)

        out["ready_to_close_target"][pos_idx] = 1.0
        out["post_close_stability_proxy"][pos_idx] = 1.0
        out["grasp_lift_proxy"][pos_idx] = float(ep_lift)
        out["planner_close_too_early"][ep_idx[early_old[ep_idx] > 0.5]] = 1.0
        out["reopen_after_trigger"][ep_idx[reopen_old[ep_idx] > 0.5]] = 1.0
        out["invalid_after_trigger"][ep_idx[invalid_old[ep_idx] > 0.5]] = 1.0
        out["fire_label_kind"][pos_idx] = 1
        out["fire_label_kind"][neg_idx] = 0

        if len(pos_idx) > 0:
            keep_indices.extend(int(x) for x in pos_idx.tolist())
        if len(neg_idx) > 0:
            keep_indices.extend(int(x) for x in neg_idx.tolist())

        total_positive += int(len(pos_idx))
        total_negative += int(len(neg_idx))
        episode_summary.append(
            {
                "episode_index": int(ep),
                "anchor_step": None if anchor_step is None else int(anchor_step),
                "positive_episode": bool(ep_lift and anchor_step is not None),
                "positive_count": int(len(pos_idx)),
                "negative_count": int(len(neg_idx)),
                "old_ready_count": int(np.sum(ep_ready)),
                "old_early_flag_count": int(np.sum(early_old[ep_idx] > 0.5)),
                "old_reopen_flag_count": int(np.sum(reopen_old[ep_idx] > 0.5)),
                "old_invalid_flag_count": int(np.sum(invalid_old[ep_idx] > 0.5)),
            }
        )

    keep_indices = np.asarray(sorted(set(keep_indices)), dtype=np.int64)
    selected = slice_rows(out, keep_indices)

    meta = {
        "source_npz": str(source_npz),
        "num_source_rows": int(num_rows),
        "num_selected_rows": int(keep_indices.shape[0]),
        "positive_count": int(total_positive),
        "negative_count": int(total_negative),
        "positive_rate": float(total_positive / max(keep_indices.shape[0], 1)),
        "positive_window": int(positive_window),
        "hard_negative_gap": int(hard_negative_gap),
        "max_negative_ratio": int(max_negative_ratio),
        "episode_summary": episode_summary,
    }
    return out, selected, meta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_npz", type=str, required=True)
    parser.add_argument("--output_root", type=str, required=True)
    parser.add_argument("--positive_window", type=int, default=1)
    parser.add_argument("--hard_negative_gap", type=int, default=8)
    parser.add_argument("--max_negative_ratio", type=int, default=8)
    parser.add_argument("--seed", type=int, default=3407)
    global args
    args = parser.parse_args()

    raw = load_npz(Path(args.input_npz))
    relabeled_rows, pose_distill_rows, meta = relabel_and_select(
        raw,
        source_npz=str(args.input_npz),
        positive_window=int(args.positive_window),
        hard_negative_gap=int(args.hard_negative_gap),
        max_negative_ratio=int(args.max_negative_ratio),
        seed=int(args.seed),
    )

    output_root = Path(args.output_root)
    save_npz(output_root / "fire_distill_rows_relabelled.npz", relabeled_rows)
    save_npz(output_root / "pose_distill_candidates.npz", pose_distill_rows)
    (output_root / "pose_distill_candidates.meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
