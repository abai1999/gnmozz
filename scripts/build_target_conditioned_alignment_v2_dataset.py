#!/usr/bin/env python3
"""Build Target-Conditioned Alignment v2 dataset from existing NPZ + cache.

Read-only. Does not modify models, refiner, or runtime code.
Outputs a self-contained NPZ with per-candidate target-relative features
and stage-aware best-action labels.
"""
from __future__ import annotations

import argparse, json
from pathlib import Path
from collections import Counter

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
XYZ_SCALE = np.array([0.008, 0.008, 0.006], dtype=np.float32)
YAW_SCALE = 0.12
K = 8

# Stage bucket thresholds
MICRO_XY, MICRO_Z, MICRO_YAW = 0.015, 0.03, 0.12
NEAR_XY, NEAR_Z, NEAR_YAW = 0.05, 0.10, 0.25
MID_XY, MID_Z = 0.12, 0.25


def _bucket(xy: float, z: float, yaw: float) -> str:
    if xy < MICRO_XY and z < MICRO_Z and yaw < MICRO_YAW:
        return "micro_contact_refine"
    if xy < NEAR_XY and z < NEAR_Z and yaw < NEAR_YAW:
        return "near_alignment"
    if xy < MID_XY and z < MID_Z:
        return "mid_approach_assist"
    return "far_coarse_approach"


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

def _best_micro(improve_xy, improve_z, improve_yaw, overshoot_xy, overshoot_z,
                overshoot_yaw, post_xy, post_z, post_yaw, action_norm):
    """Prioritize smallest post-error with overshoot penalty."""
    score = np.zeros(K, dtype=np.float64)
    w_post = np.array([2.0, 1.5, 1.0], dtype=np.float64)
    for i in range(K):
        score[i] = -(
            w_post[0] * post_xy[i] / max(MICRO_XY, 1e-6)
            + w_post[1] * post_z[i] / max(MICRO_Z, 1e-6)
            + w_post[2] * post_yaw[i] / max(MICRO_YAW, 1e-6)
            + 5.0 * (overshoot_xy[i] + overshoot_z[i] + overshoot_yaw[i])
            + 0.1 * action_norm[i]
        )
    return int(np.argmax(score))


def _best_near(improve_xy, improve_z, improve_yaw, overshoot_xy, overshoot_z,
               overshoot_yaw, post_xy, post_z, post_yaw, action_norm):
    """Prioritize positive improvement, penalize overshoot, prefer moderate action."""
    score = np.zeros(K, dtype=np.float64)
    for i in range(K):
        score[i] = (
            3.0 * np.tanh(improve_xy[i] / max(NEAR_XY, 1e-6))
            + 3.0 * np.tanh(improve_z[i] / max(NEAR_Z, 1e-6))
            + 2.0 * np.tanh(improve_yaw[i] / max(NEAR_YAW, 1e-6))
            - 5.0 * (overshoot_xy[i] + overshoot_z[i] + overshoot_yaw[i])
            - 0.05 * action_norm[i]
        )
    return int(np.argmax(score))


def _best_far(improve_xy, improve_z, improve_yaw, overshoot_xy, overshoot_z,
              overshoot_yaw, post_xy, post_z, post_yaw, action_norm):
    """Prioritize z/xy improvement, penalize large action scale risk."""
    score = np.zeros(K, dtype=np.float64)
    for i in range(K):
        score[i] = (
            1.5 * np.tanh(improve_z[i] / max(MID_Z, 1e-6))
            + 1.0 * np.tanh(improve_xy[i] / max(MID_XY, 1e-6))
            + 0.5 * np.tanh(improve_yaw[i] / max(NEAR_YAW, 1e-6))
            - 3.0 * (overshoot_xy[i] + overshoot_z[i] + overshoot_yaw[i])
            - 0.02 * action_norm[i]
        )
    return int(np.argmax(score))


# ---------------------------------------------------------------------------
# Main build
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Build v2 target-conditioned alignment dataset")
    parser.add_argument("--source_npz", type=Path, required=True)
    parser.add_argument("--cache_npz", type=Path, required=True)
    parser.add_argument("--output_npz", type=Path, required=True)
    parser.add_argument("--report_json", type=Path, required=True)
    args = parser.parse_args()

    src = np.load(args.source_npz, allow_pickle=True)
    cache = np.load(args.cache_npz, allow_pickle=True)
    N = int(src["front_rgb"].shape[0])
    print(f"[build] source rows={N}, cache proposals={cache['proposal_actions_local'].shape}")

    # --- Current-to-target delta ---
    current_to_target = np.asarray(
        src.get("proposal_target_delta_local", np.zeros((N, 6), dtype=np.float32)),
        dtype=np.float32,
    ).reshape(N, 6)

    # --- Proposals from cache (K=8) ---
    proposals = np.asarray(cache["proposal_actions_local"], dtype=np.float32).reshape(N, K, 6)

    # --- Per-candidate post-action delta and errors ---
    post_delta = current_to_target[:, None, :6] - proposals  # (N, K, 6)
    post_xy = np.linalg.norm(post_delta[:, :, :2], axis=-1)  # (N, K)
    post_z = np.abs(post_delta[:, :, 2])
    post_yaw = np.abs(post_delta[:, :, 5])

    current_xy = np.linalg.norm(current_to_target[:, :2], axis=-1)  # (N,)
    current_z = np.abs(current_to_target[:, 2])
    current_yaw = np.abs(current_to_target[:, 5])

    improve_xy = current_xy[:, None] - post_xy   # positive = improvement
    improve_z = current_z[:, None] - post_z
    improve_yaw = current_yaw[:, None] - post_yaw

    # Overshoot: post error worse than current AND action was not tiny
    action_norm = np.linalg.norm(proposals[:, :, :3], axis=-1)     # xyz action norm
    overshoot_xy = ((post_xy > current_xy[:, None]) & (action_norm > 1e-7)).astype(np.float32)
    overshoot_z = ((post_z > current_z[:, None]) & (action_norm > 1e-7)).astype(np.float32)
    overshoot_yaw = ((post_yaw > current_yaw[:, None]) & (np.abs(proposals[:, :, 5]) > 1e-7)).astype(np.float32)

    geometry_improvement = improve_xy + improve_z   # simple sum

    # --- Stage buckets ---
    buckets = np.array([_bucket(float(current_xy[i]), float(current_z[i]), float(current_yaw[i]))
                         for i in range(N)], dtype=object)
    bucket_counts = dict(Counter(buckets))

    # --- Best action labels ---
    best_micro = np.zeros(N, dtype=np.int64)
    best_near = np.zeros(N, dtype=np.int64)
    best_far = np.zeros(N, dtype=np.int64)
    best_stage = np.zeros(N, dtype=np.int64)

    for i in range(N):
        bm = _best_micro(improve_xy[i], improve_z[i], improve_yaw[i],
                         overshoot_xy[i], overshoot_z[i], overshoot_yaw[i],
                         post_xy[i], post_z[i], post_yaw[i], action_norm[i])
        bn = _best_near(improve_xy[i], improve_z[i], improve_yaw[i],
                        overshoot_xy[i], overshoot_z[i], overshoot_yaw[i],
                        post_xy[i], post_z[i], post_yaw[i], action_norm[i])
        bf = _best_far(improve_xy[i], improve_z[i], improve_yaw[i],
                       overshoot_xy[i], overshoot_z[i], overshoot_yaw[i],
                       post_xy[i], post_z[i], post_yaw[i], action_norm[i])
        best_micro[i] = bm
        best_near[i] = bn
        best_far[i] = bf
        bucket = buckets[i]
        if bucket == "micro_contact_refine":
            best_stage[i] = bm
        elif bucket == "near_alignment":
            best_stage[i] = bn
        else:
            best_stage[i] = bf

    # --- Candidate valid mask (all valid by default) ---
    candidate_valid = np.ones((N, K), dtype=np.float32)

    # --- Assemble output ---
    out = {}
    # Visual/state inputs
    for key in ("front_rgb", "wrist_rgb", "wrist_depth"):
        if key in src.files:
            out[key] = np.asarray(src[key])
    out["force_history"] = np.asarray(src.get("force_history_normalized",
                                              src.get("force_history",
                                                      src.get("ft_hist",
                                                              np.zeros((N, 32, 6), dtype=np.float32)))), dtype=np.float32)
    out["proprio"] = np.asarray(src.get("proprio", np.zeros((N, 15), dtype=np.float32)), dtype=np.float32)
    out["planner_base_action_local"] = np.asarray(
        src.get("planner_base_action_local_raw", np.zeros((N, 6), dtype=np.float32)), dtype=np.float32
    )[:, :6]

    # Target
    out["current_to_target_delta_local"] = current_to_target.astype(np.float32)

    # Candidates
    out["proposal_actions_local"] = proposals.astype(np.float32)
    out["post_candidate_delta_local"] = post_delta.astype(np.float32)

    # Error metrics
    out["current_xy_error"] = current_xy.astype(np.float32)
    out["current_z_error"] = current_z.astype(np.float32)
    out["current_yaw_error"] = current_yaw.astype(np.float32)
    out["post_xy_error"] = post_xy.astype(np.float32)
    out["post_z_error"] = post_z.astype(np.float32)
    out["post_yaw_error"] = post_yaw.astype(np.float32)
    out["xy_improvement"] = improve_xy.astype(np.float32)
    out["z_improvement"] = improve_z.astype(np.float32)
    out["yaw_improvement"] = improve_yaw.astype(np.float32)
    out["geometry_improvement"] = geometry_improvement.astype(np.float32)
    out["overshoot_xy"] = overshoot_xy.astype(np.float32)
    out["overshoot_z"] = overshoot_z.astype(np.float32)
    out["overshoot_yaw"] = overshoot_yaw.astype(np.float32)

    # Labels
    out["stage_bucket"] = buckets
    out["best_far_action_index"] = best_far
    out["best_near_action_index"] = best_near
    out["best_micro_action_index"] = best_micro
    out["best_stage_action_index"] = best_stage
    out["candidate_valid_mask"] = candidate_valid

    # IDs
    for key in ("episode_index", "step_index"):
        if key in src.files:
            out[key] = np.asarray(src[key])
    out["row_index"] = np.arange(N, dtype=np.int64)

    # Check missing
    missing = [k for k in ["front_rgb", "wrist_rgb", "wrist_depth", "force_history", "proprio"]
               if k not in out or out[k] is None]
    if missing:
        print(f"[build] WARNING: missing fields: {missing}")

    # Save
    args.output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output_npz, **out)
    print(f"[build] output_npz -> {args.output_npz} ({args.output_npz.stat().st_size / 1024 / 1024:.1f} MB)")

    # --- Report ---
    report = {
        "audit": "target_conditioned_alignment_v2_dataset_build",
        "source_npz": str(args.source_npz),
        "cache_npz": str(args.cache_npz),
        "rows": N,
        "proposal_count": K,
        "action_scale": {"xyz": XYZ_SCALE.tolist(), "yaw": YAW_SCALE},
        "stage_bucket_counts": bucket_counts,
        "missing_field_warnings": missing,
        "current_error_stats": {
            "xy": _pstats(current_xy), "z": _pstats(current_z), "yaw": _pstats(current_yaw),
        },
    }

    # Per-bucket improvement stats
    for bucket_name in ["micro_contact_refine", "near_alignment", "mid_approach_assist", "far_coarse_approach"]:
        mask = buckets == bucket_name
        n = int(mask.sum())
        if n == 0:
            report[f"improvement_{bucket_name}"] = {"n": 0, "note": "no rows in this bucket"}
            continue

        sel_idx = best_stage[mask]
        sel_improve_xy = improve_xy[mask, sel_idx]
        sel_improve_z = improve_z[mask, sel_idx]
        sel_improve_yaw = improve_yaw[mask, sel_idx]
        sel_overshoot_xy = overshoot_xy[mask, sel_idx]
        sel_overshoot_z = overshoot_z[mask, sel_idx]
        sel_post_xy = post_xy[mask, sel_idx]
        sel_post_z = post_z[mask, sel_idx]
        sel_post_yaw = post_yaw[mask, sel_idx]

        def _nonzero_rate(arr):
            return float((arr > 1e-8).mean())

        report[f"improvement_{bucket_name}"] = {
            "n": n,
            "improve_xy_mean": float(sel_improve_xy.mean()),
            "improve_z_mean": float(sel_improve_z.mean()),
            "improve_yaw_mean": float(sel_improve_yaw.mean()),
            "improve_xy_positive_rate": _nonzero_rate(sel_improve_xy),
            "improve_z_positive_rate": _nonzero_rate(sel_improve_z),
            "improve_yaw_positive_rate": _nonzero_rate(sel_improve_yaw),
            "overshoot_xy_rate": float(sel_overshoot_xy.mean()),
            "overshoot_z_rate": float(sel_overshoot_z.mean()),
            "post_xy_mean": float(sel_post_xy.mean()),
            "post_z_mean": float(sel_post_z.mean()),
            "post_yaw_mean": float(sel_post_yaw.mean()),
        }

    # Best action index histogram by bucket
    for bucket_name in ["micro_contact_refine", "near_alignment", "mid_approach_assist", "far_coarse_approach"]:
        mask = buckets == bucket_name
        idx_hist = dict(Counter(best_stage[mask].tolist()))
        report[f"best_action_hist_{bucket_name}"] = idx_hist

    # Action norm stats
    action_norms = np.linalg.norm(proposals[:, :, :3], axis=-1)
    report["candidate_action_norm_stats"] = {
        "mean": float(action_norms.mean()),
        "p50": float(np.percentile(action_norms, 50)),
        "p90": float(np.percentile(action_norms, 90)),
        "max": float(action_norms.max()),
    }

    # Action scale ratio: abs(proposal) / action_scale
    ratios_xyz = np.abs(proposals[:, :, :3]) / XYZ_SCALE.reshape(1, 1, 3)
    ratios_yaw = np.abs(proposals[:, :, 5]) / YAW_SCALE
    report["action_scale_ratio_stats"] = {
        "dx_ratio": _pstats(ratios_xyz[:, :, 0].ravel()),
        "dy_ratio": _pstats(ratios_xyz[:, :, 1].ravel()),
        "dz_ratio": _pstats(ratios_xyz[:, :, 2].ravel()),
        "dyaw_ratio": _pstats(ratios_yaw.ravel()),
    }

    # Far bucket: can fixed K=8 improve anything?
    far_mask = buckets == "far_coarse_approach"
    far_n = int(far_mask.sum())
    far_any_improve_xy = float((improve_xy[far_mask].max(axis=1) > 1e-8).mean())
    far_any_improve_z = float((improve_z[far_mask].max(axis=1) > 1e-8).mean())
    report["far_bucket_candidate_sufficiency"] = {
        "n": far_n,
        "any_candidate_improves_xy_rate": far_any_improve_xy,
        "any_candidate_improves_z_rate": far_any_improve_z,
        "verdict": (
            "sufficient_for_offline_training"
            if far_any_improve_z > 0.3 and far_any_improve_xy > 0.2
            else "insufficient_for_far_approach"
        ),
    }

    # Overall verdict
    near_n = bucket_counts.get("near_alignment", 0)
    mid_n = bucket_counts.get("mid_approach_assist", 0)
    micro_n = bucket_counts.get("micro_contact_refine", 0)
    near_mid_ok = (near_n + mid_n) > N * 0.3
    has_signal = all(
        report[f"improvement_{b}"].get("improve_z_positive_rate", 0) > 0.1
        for b in ["near_alignment", "mid_approach_assist"]
        if report[f"improvement_{b}"].get("n", 0) > 0
    )
    report["overall_verdict"] = (
        "A_sufficient_for_v2_offline_pretraining"
        if near_mid_ok and has_signal
        else "B_only_near_micro_insufficient_far" if near_mid_ok else "C_insufficient_need_resample"
    )

    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.report_json, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)
    print(f"[build] report -> {args.report_json}")

    # Print summary
    print(f"\n=== STAGE BUCKETS ===")
    for k, v in bucket_counts.items():
        print(f"  {k}: {v} ({100*v/N:.1f}%)")
    print(f"\n=== IMPROVEMENT BY BUCKET (best_stage_action) ===")
    for b in ["micro_contact_refine", "near_alignment", "mid_approach_assist", "far_coarse_approach"]:
        s = report.get(f"improvement_{b}", {})
        if s.get("n", 0) > 0:
            print(f"  {b} (n={s['n']}): xy_improve={s['improve_xy_mean']:.5f} z_improve={s['improve_z_mean']:.5f} "
                  f"yaw_improve={s['improve_yaw_mean']:.5f} z_pos_rate={s['improve_z_positive_rate']:.3f}")
    print(f"\n=== OVERALL VERDICT ===")
    print(f"  {report['overall_verdict']}")
    fs = report["far_bucket_candidate_sufficiency"]
    print(f"  far: any_z_improve={fs['any_candidate_improves_z_rate']:.3f} any_xy_improve={fs['any_candidate_improves_xy_rate']:.3f}")


def _pstats(arr):
    a = np.asarray(arr, dtype=np.float64).ravel()
    a = a[np.isfinite(a)]
    if a.size == 0:
        return {}
    return {
        "mean": float(a.mean()), "p50": float(np.percentile(a, 50)),
        "p90": float(np.percentile(a, 90)), "min": float(a.min()), "max": float(a.max()),
    }


if __name__ == "__main__":
    main()
