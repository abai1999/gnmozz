#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


INPUT_KEYS = [
    "front_rgb",
    "wrist_rgb",
    "wrist_depth",
    "proprio",
    "gripper_context",
    "proxy_current_delta_basin_target",
    "current_dx_sign",
    "current_dy_sign",
    "current_dyaw_sign",
    "basin_distance_bin",
    "substage_id",
    "contact_state",
    "stage_target_mode",
    "temporal_action_summary",
    "episode_index",
    "step_index",
    "runtime_selected_candidate_index",
    "pred_candidate_index",
    "oracle_candidate_index",
]


def _concat(paths: list[Path]) -> dict[str, np.ndarray]:
    chunks = []
    for path in paths:
        raw = np.load(path, allow_pickle=False)
        chunks.append({k: np.asarray(raw[k]) for k in raw.files})
    keys = sorted(set().union(*(c.keys() for c in chunks)))
    out: dict[str, np.ndarray] = {}
    for key in keys:
        exemplar = next((c[key] for c in chunks if key in c), None)
        if exemplar is None:
            continue
        arrs = []
        for c in chunks:
            n = int(next(iter(c.values())).shape[0])
            if key in c and tuple(np.asarray(c[key]).shape[1:]) == tuple(exemplar.shape[1:]):
                arrs.append(np.asarray(c[key]))
            else:
                shape = (n,) + tuple(exemplar.shape[1:])
                if exemplar.dtype.kind in ("U", "S", "O"):
                    arrs.append(np.full(shape, "", dtype=exemplar.dtype))
                else:
                    arrs.append(np.zeros(shape, dtype=exemplar.dtype))
        out[key] = np.concatenate(arrs, axis=0)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_npz", action="append", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--min_scope_size", type=int, default=4)
    ap.add_argument("--require_close_like", action="store_true", default=True)
    ap.add_argument("--no_require_close_like", dest="require_close_like", action="store_false")
    ap.add_argument("--teacher_xy_norm_max", type=float, default=8.0)
    ap.add_argument("--teacher_z_norm_max", type=float, default=16.0)
    ap.add_argument("--teacher_yaw_norm_max", type=float, default=3.0)
    ap.add_argument("--candidate_score_std_min", type=float, default=0.5)
    ap.add_argument("--oracle_baseline_gap_min", type=float, default=1.0)
    ap.add_argument(
        "--opportunity_policy",
        type=str,
        default="close_like_or_gap",
        choices=("close_like_only", "gap_only", "close_like_or_gap"),
    )
    ap.add_argument("--require_yaw_opportunity", action="store_true", default=False)
    ap.add_argument("--keep_yaw_abs", type=float, default=0.02)
    args = ap.parse_args()

    data = _concat([Path(p) for p in args.input_npz])
    n = int(next(iter(data.values())).shape[0])
    candidate_actions = np.asarray(data["candidate_actions_local"], dtype=np.float32)
    candidate_mask = np.asarray(data.get("candidate_mask", np.ones(candidate_actions.shape[:2], dtype=np.float32)), dtype=np.float32) > 0.5
    oracle_scores = np.asarray(data.get("candidate_oracle_score", np.zeros(candidate_actions.shape[:2], dtype=np.float32)), dtype=np.float32)
    oracle_index = np.asarray(data.get("oracle_candidate_index", np.full((n,), -1, dtype=np.int64)), dtype=np.int64)
    best_index = np.asarray(data.get("best_candidate_index", np.full((n,), -1, dtype=np.int64)), dtype=np.int64)
    baseline_index = np.asarray(
        data.get("runtime_selected_candidate_index", data.get("pred_candidate_index", np.full((n,), -1, dtype=np.int64))),
        dtype=np.int64,
    )
    teacher_metrics = np.stack(
        [
            np.asarray(data.get("teacher_truth_handoff_metric_xy_error", np.full((n,), np.nan)), dtype=np.float32),
            np.asarray(data.get("teacher_truth_handoff_metric_abs_z_error", np.full((n,), np.nan)), dtype=np.float32),
            np.asarray(data.get("teacher_truth_handoff_metric_yaw_error", np.full((n,), np.nan)), dtype=np.float32),
        ],
        axis=1,
    )
    teacher_release = np.stack(
        [
            np.asarray(data.get("teacher_truth_handoff_release_threshold_xy_error", np.full((n,), 0.0085)), dtype=np.float32),
            np.asarray(data.get("teacher_truth_handoff_release_threshold_abs_z_error", np.full((n,), 0.0035)), dtype=np.float32),
            np.asarray(data.get("teacher_truth_handoff_release_threshold_yaw_error", np.full((n,), 0.1243404)), dtype=np.float32),
        ],
        axis=1,
    )
    teacher_norm = teacher_metrics / np.maximum(teacher_release, 1e-6)

    role = np.asarray(data.get("handoff_target_role", np.full((n,), "", dtype="U64"))).astype(str)
    provider = np.asarray(data.get("target_provider_source", np.full((n,), "", dtype="U96"))).astype(str)
    close_like = (
        np.isin(role, ["pregrasp_close", "close", "commit_close"])
        | (np.char.find(provider, "canonical_close_orientation_contract") >= 0)
        | (np.char.find(provider, "teacher_motion") >= 0)
    )

    scope_size = np.sum(candidate_mask, axis=1)
    score_std = np.nanstd(oracle_scores, axis=1)
    keep = np.all(np.isfinite(candidate_actions[:, :, :6]), axis=(1, 2))
    keep &= scope_size >= int(args.min_scope_size)
    keep &= np.all(np.isfinite(teacher_norm), axis=1)
    keep &= teacher_norm[:, 0] <= float(args.teacher_xy_norm_max)
    keep &= teacher_norm[:, 1] <= float(args.teacher_z_norm_max)
    keep &= teacher_norm[:, 2] <= float(args.teacher_yaw_norm_max)
    keep &= np.isfinite(score_std)
    keep &= score_std >= float(args.candidate_score_std_min)

    # Prefer oracle label, then teacher-best label.
    best_label = np.where(oracle_index >= 0, oracle_index, best_index).astype(np.int64)
    keep &= best_label >= 0
    keep &= best_label < candidate_actions.shape[1]
    keep &= baseline_index >= 0
    keep &= baseline_index < candidate_actions.shape[1]

    row_ids = np.arange(n, dtype=np.int64)
    best_scores_full = oracle_scores[row_ids, np.clip(best_label, 0, oracle_scores.shape[1] - 1)]
    baseline_scores_full = oracle_scores[row_ids, np.clip(baseline_index, 0, oracle_scores.shape[1] - 1)]
    oracle_baseline_gap = best_scores_full - baseline_scores_full
    has_gap = oracle_baseline_gap >= float(args.oracle_baseline_gap_min)
    candidate_yaw_nonzero = np.any((np.abs(candidate_actions[:, :, 5]) > float(args.keep_yaw_abs)) & candidate_mask, axis=1)
    improving_yaw = np.any(
        (np.abs(candidate_actions[:, :, 5]) > float(args.keep_yaw_abs))
        & candidate_mask
        & (oracle_scores > (baseline_scores_full[:, None] + 1e-6)),
        axis=1,
    )

    if args.opportunity_policy == "close_like_only":
        opportunity_mask = close_like
    elif args.opportunity_policy == "gap_only":
        opportunity_mask = has_gap
    else:
        opportunity_mask = close_like | has_gap
    if bool(args.require_close_like):
        keep &= opportunity_mask
    else:
        keep &= opportunity_mask
    if bool(args.require_yaw_opportunity):
        keep &= improving_yaw

    idx = np.where(keep)[0]
    if idx.size == 0:
        raise RuntimeError("no candidate-ranking rows survived filtering")

    out: dict[str, np.ndarray] = {}
    for key in INPUT_KEYS:
        if key in data:
            out[key] = np.asarray(data[key])[idx]
    out["candidate_actions_local"] = candidate_actions[idx].astype(np.float32)
    out["candidate_mask"] = candidate_mask[idx].astype(np.float32)
    out["candidate_oracle_score"] = oracle_scores[idx].astype(np.float32)
    out["candidate_best_index"] = best_label[idx].astype(np.int64)
    out["candidate_baseline_index"] = baseline_index[idx].astype(np.int64)
    out["candidate_teacher_norm"] = teacher_norm[idx].astype(np.float32)
    out["candidate_scope_size"] = scope_size[idx].astype(np.float32)
    out["sample_weight"] = (
        1.0
        + 1.0 * (teacher_norm[idx, 0] <= 1.0).astype(np.float32)
        + 1.0 * (teacher_norm[idx, 1] <= 1.0).astype(np.float32)
        + 1.0 * (teacher_norm[idx, 2] <= 1.0).astype(np.float32)
    ).astype(np.float32)

    filtered_scores = oracle_scores[idx].astype(np.float32)
    filtered_best = best_label[idx].astype(np.int64)
    filtered_baseline = baseline_index[idx].astype(np.int64)
    row_ids = np.arange(idx.size, dtype=np.int64)
    best_scores = filtered_scores[row_ids, filtered_best]
    baseline_scores = filtered_scores[row_ids, filtered_baseline]
    filtered_score_std = np.nanstd(filtered_scores, axis=1)
    yaw_nonzero_ratio = np.mean(np.abs(candidate_actions[idx, :, 5]) > 1e-4, axis=1).astype(np.float32)
    best_is_yaw = (np.abs(candidate_actions[idx, filtered_best, 5]) > 1e-4).astype(np.float32)
    baseline_diff = (filtered_best != filtered_baseline).astype(np.float32)
    filtered_improving_yaw = improving_yaw[idx].astype(np.float32)
    filtered_close_like = close_like[idx].astype(np.float32)
    filtered_gap = oracle_baseline_gap[idx].astype(np.float32)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "alignment_v4_candidate_ranking_dataset.npz"
    np.savez_compressed(out_path, **out)

    report = {
        "rows": int(idx.size),
        "input_rows": int(n),
        "episodes": int(np.unique(out["episode_index"]).size) if "episode_index" in out else 0,
        "mean_scope_size": float(np.mean(out["candidate_scope_size"])),
        "teacher_xy_norm_mean": float(np.mean(out["candidate_teacher_norm"][:, 0])),
        "teacher_z_norm_mean": float(np.mean(out["candidate_teacher_norm"][:, 1])),
        "teacher_yaw_norm_mean": float(np.mean(out["candidate_teacher_norm"][:, 2])),
        "score_std_mean": float(np.nanmean(filtered_score_std)),
        "oracle_baseline_gap_mean": float(np.nanmean(best_scores - baseline_scores)),
        "candidate_yaw_nonzero_ratio": float(np.nanmean(yaw_nonzero_ratio)),
        "best_is_yaw_candidate_ratio": float(np.nanmean(best_is_yaw)),
        "baseline_to_oracle_different_ratio": float(np.nanmean(baseline_diff)),
        "improving_yaw_row_ratio": float(np.nanmean(filtered_improving_yaw)),
        "close_like_row_ratio": float(np.nanmean(filtered_close_like)),
        "oracle_baseline_gap_p50": float(np.nanpercentile(filtered_gap, 50)),
        "oracle_baseline_gap_p90": float(np.nanpercentile(filtered_gap, 90)),
        "opportunity_policy": str(args.opportunity_policy),
        "candidate_score_std_min": float(args.candidate_score_std_min),
        "oracle_baseline_gap_min": float(args.oracle_baseline_gap_min),
        "keep_yaw_abs": float(args.keep_yaw_abs),
        "per_episode_rows": {
            str(int(ep)): int(np.sum(out["episode_index"] == ep))
            for ep in np.unique(out["episode_index"])
        }
        if "episode_index" in out
        else {},
        "output_npz": str(out_path),
    }
    (out_dir / "alignment_v4_candidate_ranking_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
