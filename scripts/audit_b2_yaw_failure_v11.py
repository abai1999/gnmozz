#!/usr/bin/env python3
"""Audit B2 yaw-mode failure modes from a v11/v12 yaw-mode dataset.

This is a read-only diagnostic script.  It summarizes whether yaw mode labels
are limited by candidate-bank coverage, source/profile confounding, or model
separability.  It does not touch runtime/provider code.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


BIN_NAMES = ["no_yaw", "small_neg", "small_pos", "large_neg", "large_pos"]


def _source_profile(source: str) -> str:
    s = source.lower()
    if "oracle" in s or "oracleub" in s:
        return "oracle"
    if "teacher_assisted" in s:
        return "teacher_assisted"
    if "yawapply" in s:
        return "targeted_yawapply"
    if "learned" in s or "lateprofile" in s or "recollect" in s or "runtime" in s or "current_profile" in s:
        return "runtime_like"
    return "other"


def _counts(values) -> dict[str, int]:
    if len(values) == 0:
        return {}
    keys, vals = np.unique(values, return_counts=True)
    return {str(k): int(v) for k, v in zip(keys, vals)}


def _dominant_profile(profiles: np.ndarray) -> tuple[str, float]:
    if profiles.size == 0:
        return "none", 0.0
    counts = _counts(profiles)
    k = max(counts, key=counts.get)
    return str(k), float(counts[k] / max(int(profiles.size), 1))


def _stats(x: np.ndarray) -> dict[str, float | None]:
    x = np.asarray(x, dtype=np.float32)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {"mean": None, "p10": None, "p50": None, "p90": None}
    return {
        "mean": float(np.mean(x)),
        "p10": float(np.percentile(x, 10)),
        "p50": float(np.percentile(x, 50)),
        "p90": float(np.percentile(x, 90)),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_npz", required=True)
    ap.add_argument("--output_json", required=True)
    ap.add_argument("--focus_episodes", default="12,23,45")
    ap.add_argument("--confound_dominance", type=float, default=0.60)
    args = ap.parse_args()

    data = np.load(args.dataset_npz, allow_pickle=False)
    ep = np.asarray(data["episode_index"], dtype=np.int64)
    source = np.asarray(data.get("source_name", np.full((ep.shape[0],), "unknown", dtype="U32"))).astype(str)
    profiles = np.asarray([_source_profile(s) for s in source], dtype="U32")
    yaw_needed = np.asarray(data.get("yaw_needed_v1", np.zeros((ep.shape[0],), dtype=np.float32))) > 0.5
    label = np.asarray(data.get("yaw_mode3_label_v11", np.full((ep.shape[0],), -1, dtype=np.int64)), dtype=np.int64)
    valid = np.asarray(data.get("yaw_mode_valid_v11", label >= 0), dtype=np.float32) > 0.5
    counts = np.asarray(data.get("b2_yaw_cost_curve_counts_v11"), dtype=np.int64)
    scores = np.asarray(data.get("b2_yaw_cost_curve_scores_v11"), dtype=np.float32)
    adv = np.asarray(data.get("yaw_advantage_cont_v11", np.zeros((ep.shape[0],), dtype=np.float32)), dtype=np.float32)

    has_no = counts[:, 0] > 0
    has_neg = (counts[:, 1] + counts[:, 3]) > 0
    has_pos = (counts[:, 2] + counts[:, 4]) > 0
    complete = yaw_needed & has_no & has_neg & has_pos

    episode = {}
    source_confounded_eps = []
    for e in sorted(np.unique(ep[yaw_needed]).astype(int).tolist()):
        m = (ep == e) & yaw_needed
        keep = m & valid & (label == 0)
        small = m & valid & (label == 1)
        apply = m & valid & (label == 2)
        keep_dom, keep_frac = _dominant_profile(profiles[keep])
        apply_dom, apply_frac = _dominant_profile(profiles[apply])
        clean_same_profile = False
        for p in sorted(set(profiles[m].astype(str).tolist())):
            pm = m & (profiles.astype(str) == p)
            if np.any(pm & keep) and np.any(pm & apply):
                clean_same_profile = True
                break
        source_confounded = bool(
            np.any(keep)
            and np.any(apply)
            and keep_dom != apply_dom
            and keep_frac >= float(args.confound_dominance)
            and apply_frac >= float(args.confound_dominance)
            and not clean_same_profile
        )
        if source_confounded:
            source_confounded_eps.append(int(e))
        episode[str(e)] = {
            "yaw_needed_rows": int(np.sum(m)),
            "keep_rows": int(np.sum(keep)),
            "small_rows": int(np.sum(small)),
            "apply_rows": int(np.sum(apply)),
            "candidate_complete_rows": int(np.sum(complete & (ep == e))),
            "candidate_complete_rate": float(np.mean(complete[m])) if np.any(m) else 0.0,
            "bin_available_rows": {name: int(np.sum(m & (counts[:, i] > 0))) for i, name in enumerate(BIN_NAMES)},
            "source_counts": _counts(source[m]),
            "profile_counts": _counts(profiles[m]),
            "keep_profile_counts": _counts(profiles[keep]),
            "apply_profile_counts": _counts(profiles[apply]),
            "keep_dominant_profile": keep_dom,
            "apply_dominant_profile": apply_dom,
            "source_confounded": source_confounded,
            "yaw_advantage": _stats(adv[m]),
            "curve_score_means": {
                name: (float(np.nanmean(np.where(np.isfinite(scores[m, i]), scores[m, i], np.nan))) if np.any(m) else None)
                for i, name in enumerate(BIN_NAMES)
            },
        }

    focus = [x.strip() for x in args.focus_episodes.split(",") if x.strip()]
    result = {
        "dataset_npz": str(args.dataset_npz),
        "rows": int(ep.shape[0]),
        "yaw_needed_rows": int(np.sum(yaw_needed)),
        "yaw_needed_eps": int(np.unique(ep[yaw_needed]).size),
        "candidate_complete_rows": int(np.sum(complete)),
        "candidate_complete_rate": float(np.mean(complete[yaw_needed])) if np.any(yaw_needed) else 0.0,
        "source_confounded_eps": source_confounded_eps,
        "focus_episodes": {e: episode.get(e, {}) for e in focus},
        "episodes": episode,
        "diagnosis": {
            "candidate_space_blocker": bool(float(np.mean(complete[yaw_needed])) < 0.95) if np.any(yaw_needed) else True,
            "source_confounding_blocker": bool(len(source_confounded_eps) > 0),
            "runtime_policy_changed": False,
        },
    }
    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(json.dumps(result["diagnosis"], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
