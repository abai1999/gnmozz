#!/usr/bin/env python3
"""Build B2 yaw-mode v11/v12/v13 diagnostics and labels.

V11 is deliberately a semantic cleanup layer, not a runtime change.  It keeps
the existing candidate bank and adds:

- a yaw cost curve over no-yaw / small-yaw / large-yaw direction bins;
- continuous yaw advantage relative to no-yaw;
- 3-way mode labels: keep, small-yaw, apply-yaw;
- confidence, manifest, gate report, and per-episode audit.

V13 keeps the same curve semantics but can restrict supervised mode labels to
current-profile rows.  Oracle/teacher-assisted rows remain available for
window-level diagnostics and cost-curve audits, but they do not dominate the
apply/keep supervision used for B2.

Scores follow the existing convention: higher candidate_oracle_score is better.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


BIN_NAMES = ("no_yaw", "small_neg", "small_pos", "large_neg", "large_pos")


def _json_float(x) -> float | None:
    x = float(x)
    return x if np.isfinite(x) else None


def _episode_counts(ep: np.ndarray, mask: np.ndarray) -> dict:
    rows = int(np.sum(mask))
    eps = sorted(int(x) for x in np.unique(ep[mask]).tolist()) if rows else []
    return {"rows": rows, "eps": len(eps), "episode_indices": eps, "low_confidence": rows < 25}


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


def _dominant_profile(profiles: np.ndarray) -> tuple[str, float]:
    if profiles.size == 0:
        return "none", 0.0
    keys, vals = np.unique(profiles, return_counts=True)
    i = int(np.argmax(vals))
    return str(keys[i]), float(vals[i] / max(int(np.sum(vals)), 1))


def _best_in_bin(scores: np.ndarray, mask: np.ndarray, actions_yaw: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = scores.shape[0]
    best_score = np.full((n,), -np.inf, dtype=np.float32)
    best_idx = np.full((n,), -1, dtype=np.int64)
    best_yaw = np.full((n,), np.nan, dtype=np.float32)
    for i in range(n):
        idx = np.flatnonzero(mask[i])
        if idx.size == 0:
            continue
        local = scores[i, idx]
        j = int(idx[int(np.argmax(local))])
        best_score[i] = float(scores[i, j])
        best_idx[i] = j
        best_yaw[i] = float(actions_yaw[i, j])
    return best_score, best_idx, best_yaw


def _pick_split(
    ep: np.ndarray,
    keep: np.ndarray,
    small: np.ndarray,
    apply: np.ndarray,
    seed: int,
    val_ratio: float,
    min_train_apply_eps: int,
    min_train_keep_eps: int,
) -> dict:
    rng = np.random.default_rng(seed)
    uniq = np.unique(ep)
    val_n = max(1, int(round(uniq.size * val_ratio)))
    if val_n >= uniq.size:
        val_n = max(1, uniq.size - 1)

    def eps_for(mask: np.ndarray) -> list[int]:
        xs = np.unique(ep[mask]).astype(np.int64)
        rng.shuffle(xs)
        return [int(x) for x in xs.tolist()]

    keep_eps = eps_for(keep)
    small_eps = eps_for(small)
    apply_eps = eps_for(apply)
    val: list[int] = []
    apply_set = set(apply_eps)
    keep_set = set(keep_eps)

    def add(candidates: list[int], need: int):
        for e in candidates:
            if len([x for x in val if x in candidates]) >= need:
                break
            if len(val) >= val_n:
                break
            if e not in val:
                val.append(e)

    # First reserve validation slots for the scarce mode episodes.  Prefer
    # distinct keep episodes that are not already selected for apply so the val
    # split tests both semantics across episodes rather than one mixed episode.
    add(apply_eps, min(2, max(len(apply_eps) - 2, 0), val_n))
    keep_priority = (
        [e for e in keep_eps if e not in val and e not in apply_set]
        + [e for e in keep_eps if e not in val]
        + [e for e in keep_eps if e in val]
    )
    add(keep_priority, min(3, max(len(keep_eps) - 3, 0), val_n))
    add(small_eps, min(1, max(len(small_eps) - 1, 0), val_n))
    shuffled = uniq.copy()
    rng.shuffle(shuffled)
    for e in shuffled.tolist():
        if len(val) >= val_n:
            break
        if int(e) not in val:
            val.append(int(e))
    # Do not let stratified validation consume the scarce apply/keep episode
    # diversity needed for training. Drop least-essential validation episodes
    # until the train side keeps its minimum coverage, but try not to drop below
    # the requested validation mode coverage if an alternative non-mode episode
    # can be removed instead.
    changed = True
    while changed:
        changed = False
        train_apply = len(apply_set - set(val))
        train_keep = len(keep_set - set(val))
        if train_apply < int(min_train_apply_eps):
            for e in list(reversed(val)):
                if e in apply_set:
                    val.remove(e)
                    changed = True
                    break
        if train_keep < int(min_train_keep_eps):
            for e in list(reversed(val)):
                if e in keep_set:
                    val.remove(e)
                    changed = True
                    break
    # Final repair pass: if validation mode coverage is still short and the
    # training side can spare an episode, swap out non-mode validation episodes.
    def ensure_val_coverage(candidates: list[int], target: int, train_min: int) -> None:
        nonlocal val
        cand_set = set(candidates)
        while len(cand_set & set(val)) < int(target):
            train_count = len(cand_set - set(val))
            if train_count <= int(train_min):
                break
            add_ep = next((e for e in candidates if e not in val), None)
            if add_ep is None:
                break
            remove_ep = next((e for e in val if e not in apply_set and e not in keep_set and e not in set(small_eps)), None)
            if remove_ep is None and len(val) >= val_n:
                # Last resort: remove an episode that does not belong to this
                # target set, while preserving the other target set when possible.
                remove_ep = next((e for e in val if e not in cand_set), None)
            if remove_ep is not None:
                val.remove(remove_ep)
            elif len(val) >= val_n:
                break
            if add_ep not in val:
                val.append(add_ep)

    ensure_val_coverage(apply_eps, min(2, max(len(apply_eps) - int(min_train_apply_eps), 0)), min_train_apply_eps)
    ensure_val_coverage(keep_eps, min(3, max(len(keep_eps) - int(min_train_keep_eps), 0)), min_train_keep_eps)
    # If a keep repair added a mixed apply+keep episode, restore the train-side
    # apply minimum by removing the least essential validation apply episode.
    for candidates, train_min in ((apply_eps, min_train_apply_eps), (keep_eps, min_train_keep_eps)):
        cand_set = set(candidates)
        while len(cand_set - set(val)) < int(train_min):
            remove_ep = next((e for e in reversed(val) if e in cand_set), None)
            if remove_ep is None:
                break
            val.remove(remove_ep)
    val_set = set(val)
    split = np.zeros((ep.shape[0],), dtype=np.int64)
    split[np.isin(ep, list(val_set))] = 1
    train = split == 0
    val_mask = split == 1
    return {
        "split_v11": split,
        "train_episodes": int(np.unique(ep[train]).size),
        "val_episodes": int(np.unique(ep[val_mask]).size),
        "val_episode_indices": sorted(int(x) for x in val_set),
        "train_yaw_apply_eps": int(np.unique(ep[train & apply]).size),
        "val_yaw_apply_eps": int(np.unique(ep[val_mask & apply]).size),
        "train_yaw_keep_eps": int(np.unique(ep[train & keep]).size),
        "val_yaw_keep_eps": int(np.unique(ep[val_mask & keep]).size),
        "train_yaw_small_eps": int(np.unique(ep[train & small]).size),
        "val_yaw_small_eps": int(np.unique(ep[val_mask & small]).size),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_npz", required=True)
    ap.add_argument("--output_npz", required=True)
    ap.add_argument("--manifest_json", required=True)
    ap.add_argument("--gate_report_json", required=True)
    ap.add_argument("--episode_audit_json", required=True)
    ap.add_argument("--focus_episodes", default="12,23,45")
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--val_ratio", type=float, default=0.25)
    ap.add_argument("--keep_yaw_abs", type=float, default=0.035)
    ap.add_argument("--large_yaw_abs", type=float, default=0.075)
    ap.add_argument("--keep_margin", type=float, default=0.25)
    ap.add_argument("--small_margin", type=float, default=0.35)
    ap.add_argument("--apply_margin", type=float, default=0.75)
    ap.add_argument("--large_over_small_margin", type=float, default=0.15)
    ap.add_argument("--confidence_scale", type=float, default=1.0)
    ap.add_argument("--min_yaw_apply_eps", type=int, default=6)
    ap.add_argument("--min_yaw_keep_eps", type=int, default=6)
    ap.add_argument("--min_val_yaw_apply_eps", type=int, default=2)
    ap.add_argument("--min_val_yaw_keep_eps", type=int, default=3)
    ap.add_argument("--min_train_yaw_apply_eps", type=int, default=4)
    ap.add_argument("--min_train_yaw_keep_eps", type=int, default=4)
    ap.add_argument("--schema_name", default="b2_yawmode_dataset_v11")
    ap.add_argument("--compressed", action="store_true")
    ap.add_argument("--require_complete_yaw_bank", action="store_true")
    ap.add_argument("--source_confound_dominance", type=float, default=0.60)
    ap.add_argument(
        "--supervised_profiles",
        default="",
        help=(
            "Comma-separated source_profile values allowed to contribute yaw mode labels. "
            "Empty means all profiles. For v13 current-profile gates use e.g. "
            "'runtime_like,targeted_yawapply'."
        ),
    )
    ap.add_argument(
        "--diagnostic_profiles",
        default="oracle,teacher_assisted",
        help="Profiles kept for curve diagnostics but usually excluded from supervised yaw labels.",
    )
    ap.add_argument(
        "--min_profile_yaw_apply_eps",
        type=int,
        default=0,
        help="If >0, require at least this many apply episodes inside one supervised profile.",
    )
    ap.add_argument(
        "--min_profile_yaw_keep_eps",
        type=int,
        default=0,
        help="If >0, require at least this many keep episodes inside one supervised profile.",
    )
    ap.add_argument(
        "--mode_semantics",
        default="v11_curve3way",
        choices=("v11_curve3way", "v14_observable_binary", "v14b_observable_summary_binary"),
        help=(
            "How to convert the teacher yaw curve into supervised yaw-mode labels. "
            "v11_curve3way keeps the original keep/small/apply definition. "
            "v14_observable_binary keeps only hard keep/apply anchors that are "
            "also consistent with runtime-visible yaw observables. "
            "v14b_observable_summary_binary replaces the single proxy-dyaw cut "
            "with candidate-summary observable anchors."
        ),
    )
    ap.add_argument(
        "--v14_keep_proxy_yaw_abs_max",
        type=float,
        default=0.16,
        help="Maximum |proxy dyaw| retained as a hard v14 keep anchor.",
    )
    ap.add_argument(
        "--v14_apply_proxy_yaw_abs_min",
        type=float,
        default=0.18,
        help="Minimum |proxy dyaw| retained as a hard v14 apply anchor.",
    )
    ap.add_argument(
        "--v14_apply_require_sign_match",
        action="store_true",
        help="Require v14 apply anchors to have the same yaw sign as proxy_current_delta_basin_target.",
    )
    ap.add_argument(
        "--v14_abstain_ambiguous",
        action="store_true",
        help="When using v14 semantics, leave non-anchor rows out of supervised mode CE.",
    )
    ap.add_argument(
        "--drop_source_confounded_supervision",
        action="store_true",
        help="Drop source-confounded episodes from supervised yaw-mode labels after audit.",
    )
    ap.add_argument("--v14b_keep_no_yaw_frac_min", type=float, default=0.48)
    ap.add_argument("--v14b_keep_scope_size_min", type=float, default=20.0)
    ap.add_argument("--v14b_keep_abs_mean_max", type=float, default=0.060)
    ap.add_argument("--v14b_keep_std_max", type=float, default=0.078)
    ap.add_argument("--v14b_keep_soft_no_yaw_frac_min", type=float, default=0.40)
    ap.add_argument("--v14b_keep_tight_abs_mean_max", type=float, default=0.055)
    ap.add_argument("--v14b_keep_tight_std_max", type=float, default=0.075)
    ap.add_argument("--v14b_apply_no_yaw_frac_max", type=float, default=0.40)
    ap.add_argument("--v14b_apply_large_frac_min", type=float, default=0.42)
    ap.add_argument("--v14b_apply_scope_size_max", type=float, default=16.5)
    ap.add_argument("--v14b_apply_abs_mean_min", type=float, default=0.068)
    ap.add_argument("--v14b_apply_std_min", type=float, default=0.082)
    ap.add_argument("--v14b_apply_sign_agree_min", type=float, default=0.62)
    args = ap.parse_args()

    arr = np.load(args.input_npz, allow_pickle=False)
    data = {k: np.asarray(arr[k]) for k in arr.files}
    required = ["episode_index", "candidate_actions_local", "candidate_mask", "candidate_oracle_score"]
    missing = [k for k in required if k not in data]
    if missing:
        raise RuntimeError(f"missing required fields: {missing}")

    ep = np.asarray(data["episode_index"], dtype=np.int64)
    actions = np.asarray(data["candidate_actions_local"], dtype=np.float32)
    yaw_action = actions[:, :, 5]
    scores = np.asarray(data["candidate_oracle_score"], dtype=np.float32)
    mask = np.asarray(data["candidate_mask"], dtype=np.float32) > 0.5
    scope = np.asarray(data.get("b2_yaw_aware_candidate_scope_v3", data["candidate_mask"]), dtype=np.float32) > 0.5
    scope &= mask
    yaw_needed = np.asarray(data.get("yaw_needed_v1", np.zeros((ep.shape[0],), dtype=np.float32)), dtype=np.float32) > 0.5
    source = np.asarray(data.get("source_name", np.full((ep.shape[0],), "unknown", dtype="U32"))).astype("U64")
    source_profile = np.asarray([_source_profile(str(s)) for s in source], dtype="U32")
    supervised_profiles = {x.strip() for x in str(args.supervised_profiles).split(",") if x.strip()}
    diagnostic_profiles = {x.strip() for x in str(args.diagnostic_profiles).split(",") if x.strip()}
    if supervised_profiles:
        supervised_profile_mask = np.isin(source_profile.astype(str), sorted(supervised_profiles))
    else:
        supervised_profile_mask = np.ones((ep.shape[0],), dtype=bool)
    diagnostic_profile_mask = np.isin(source_profile.astype(str), sorted(diagnostic_profiles))

    no = scope & (np.abs(yaw_action) <= float(args.keep_yaw_abs))
    small = scope & (np.abs(yaw_action) > float(args.keep_yaw_abs)) & (np.abs(yaw_action) <= float(args.large_yaw_abs))
    large = scope & (np.abs(yaw_action) > float(args.large_yaw_abs))
    bin_masks = [
        no,
        small & (yaw_action < 0.0),
        small & (yaw_action > 0.0),
        large & (yaw_action < 0.0),
        large & (yaw_action > 0.0),
    ]
    curve_scores, curve_idx, curve_yaw, curve_counts = [], [], [], []
    for bm in bin_masks:
        bs, bi, by = _best_in_bin(scores, bm, yaw_action)
        curve_scores.append(bs)
        curve_idx.append(bi)
        curve_yaw.append(by)
        curve_counts.append(np.sum(bm, axis=1).astype(np.int64))
    curve_scores = np.stack(curve_scores, axis=1).astype(np.float32)
    curve_idx = np.stack(curve_idx, axis=1).astype(np.int64)
    curve_yaw = np.stack(curve_yaw, axis=1).astype(np.float32)
    curve_counts = np.stack(curve_counts, axis=1).astype(np.int64)

    no_score = curve_scores[:, 0]
    small_score = np.max(curve_scores[:, 1:3], axis=1)
    large_score = np.max(curve_scores[:, 3:5], axis=1)
    yaw_score = np.max(curve_scores[:, 1:5], axis=1)
    no_valid = np.isfinite(no_score)
    small_valid = np.isfinite(small_score)
    large_valid = np.isfinite(large_score)
    yaw_valid = np.isfinite(yaw_score)
    yaw_advantage = np.where(no_valid & yaw_valid, yaw_score - no_score, 0.0).astype(np.float32)
    small_advantage = np.where(no_valid & small_valid, small_score - no_score, -np.inf)
    large_advantage = np.where(no_valid & large_valid, large_score - no_score, -np.inf)
    apply_advantage = np.where(
        no_valid & large_valid,
        large_score - np.maximum(no_score, np.where(small_valid, small_score, -np.inf)),
        -np.inf,
    )
    proxy_delta = np.asarray(data.get("proxy_current_delta_basin_target", np.zeros((ep.shape[0], 6), dtype=np.float32)), dtype=np.float32)
    proxy_dyaw = proxy_delta[:, 5] if proxy_delta.ndim == 2 and proxy_delta.shape[1] >= 6 else np.zeros((ep.shape[0],), dtype=np.float32)
    proxy_dyaw_abs = np.abs(proxy_dyaw)

    label = np.full((ep.shape[0],), -1, dtype=np.int64)
    confidence = np.zeros((ep.shape[0],), dtype=np.float32)
    valid = yaw_needed & no_valid & yaw_valid

    keep_curve = valid & supervised_profile_mask & (yaw_advantage <= float(args.keep_margin))
    apply_curve = (
        valid
        & supervised_profile_mask
        & (large_advantage >= float(args.apply_margin))
        & (apply_advantage >= float(args.large_over_small_margin))
    )
    scope_count = np.sum(scope, axis=1).astype(np.float32)
    safe_scope = np.maximum(scope_count, 1.0)
    yaw_abs = np.abs(yaw_action)
    no_frac = np.sum(scope & (yaw_abs <= float(args.keep_yaw_abs)), axis=1).astype(np.float32) / safe_scope
    small_frac = np.sum(
        scope & (yaw_abs > float(args.keep_yaw_abs)) & (yaw_abs <= float(args.large_yaw_abs)),
        axis=1,
    ).astype(np.float32) / safe_scope
    large_frac = np.sum(scope & (yaw_abs > float(args.large_yaw_abs)), axis=1).astype(np.float32) / safe_scope
    pos_frac = np.sum(scope & (yaw_action > 1e-6), axis=1).astype(np.float32) / safe_scope
    neg_frac = np.sum(scope & (yaw_action < -1e-6), axis=1).astype(np.float32) / safe_scope
    sum_abs = np.sum(np.where(scope, yaw_abs, 0.0), axis=1).astype(np.float32)
    yaw_abs_mean = sum_abs / safe_scope
    yaw_mean = np.sum(np.where(scope, yaw_action, 0.0), axis=1).astype(np.float32) / safe_scope
    yaw_var = np.sum(np.where(scope, (yaw_action - yaw_mean[:, None]) ** 2, 0.0), axis=1).astype(np.float32) / safe_scope
    yaw_std = np.sqrt(np.maximum(yaw_var, 0.0)).astype(np.float32)
    proxy_sign = np.sign(proxy_dyaw).astype(np.float32)
    sign_agree_frac = np.where(
        np.abs(proxy_sign) > 1e-6,
        np.where(proxy_sign > 0.0, pos_frac, neg_frac),
        0.0,
    ).astype(np.float32)
    if str(args.mode_semantics) == "v14_observable_binary":
        best_large_sign = np.sign(np.where(np.isfinite(curve_yaw[:, 3]), curve_yaw[:, 3], 0.0) + np.where(np.isfinite(curve_yaw[:, 4]), curve_yaw[:, 4], 0.0))
        sign_match = (best_large_sign == proxy_sign) | (best_large_sign == 0.0) | (proxy_sign == 0.0)
        keep = keep_curve & (proxy_dyaw_abs <= float(args.v14_keep_proxy_yaw_abs_max))
        apply = apply_curve & (proxy_dyaw_abs >= float(args.v14_apply_proxy_yaw_abs_min))
        if bool(args.v14_apply_require_sign_match):
            apply &= sign_match
        ambiguous = valid & supervised_profile_mask & ~(keep | apply)
        small_mode = np.zeros_like(valid, dtype=bool) if bool(args.v14_abstain_ambiguous) else ambiguous.copy()
    elif str(args.mode_semantics) == "v14b_observable_summary_binary":
        observable_keep = (
            (no_frac >= float(args.v14b_keep_no_yaw_frac_min))
            | (scope_count >= float(args.v14b_keep_scope_size_min))
            | (
                (no_frac >= float(args.v14b_keep_soft_no_yaw_frac_min))
                & (yaw_abs_mean <= float(args.v14b_keep_abs_mean_max))
                & (yaw_std <= float(args.v14b_keep_std_max))
            )
            | (
                (yaw_abs_mean <= float(args.v14b_keep_tight_abs_mean_max))
                & (yaw_std <= float(args.v14b_keep_tight_std_max))
            )
        )
        observable_apply = (
            (no_frac <= float(args.v14b_apply_no_yaw_frac_max))
            & (large_frac >= float(args.v14b_apply_large_frac_min))
            & (scope_count <= float(args.v14b_apply_scope_size_max))
            & (yaw_abs_mean >= float(args.v14b_apply_abs_mean_min))
            & (yaw_std >= float(args.v14b_apply_std_min))
            & (sign_agree_frac >= float(args.v14b_apply_sign_agree_min))
        )
        keep = keep_curve & observable_keep
        apply = apply_curve & observable_apply
        ambiguous = valid & supervised_profile_mask & ~(keep | apply)
        small_mode = np.zeros_like(valid, dtype=bool) if bool(args.v14_abstain_ambiguous) else ambiguous.copy()
    else:
        keep = keep_curve
        apply = apply_curve
        # Middle rows are not strong enough to force a large-yaw apply, but also
        # not clean no-op anchors. Treat them as small-yaw/uncertain with low
        # confidence instead of silently removing them from mode supervision.
        small_mode = valid & supervised_profile_mask & ~keep & ~apply
        ambiguous = np.zeros_like(valid, dtype=bool)

    label[keep] = 0
    label[small_mode] = 1
    label[apply] = 2
    conf_scale = max(float(args.confidence_scale), 1e-6)
    confidence[keep] = np.clip((float(args.keep_margin) - yaw_advantage[keep]) / conf_scale, 0.05, 1.0)
    middle_adv = np.where(np.isfinite(small_advantage), small_advantage, yaw_advantage)
    confidence[small_mode] = np.clip(np.abs(middle_adv[small_mode]) / conf_scale, 0.05, 0.5)
    confidence[apply] = np.clip(large_advantage[apply] / conf_scale, 0.05, 1.0)
    if str(args.mode_semantics) in {"v14_observable_binary", "v14b_observable_summary_binary"}:
        confidence[keep] *= np.clip(
            (
                (float(args.v14_keep_proxy_yaw_abs_max) - proxy_dyaw_abs[keep])
                / max(float(args.v14_keep_proxy_yaw_abs_max), 1e-6)
                if str(args.mode_semantics) == "v14_observable_binary"
                else np.maximum(
                    no_frac[keep] - float(args.v14b_keep_soft_no_yaw_frac_min),
                    (scope_count[keep] - float(args.v14b_apply_scope_size_max)) / max(float(args.v14b_keep_scope_size_min), 1.0),
                )
            ),
            0.25,
            1.0,
        )
        confidence[apply] *= np.clip(
            (
                (proxy_dyaw_abs[apply] - float(args.v14_apply_proxy_yaw_abs_min)) / max(0.05, float(args.v14_apply_proxy_yaw_abs_min))
                if str(args.mode_semantics) == "v14_observable_binary"
                else np.minimum.reduce(
                    [
                        np.maximum(large_frac[apply] - float(args.v14b_apply_large_frac_min), 0.0) / max(1.0 - float(args.v14b_apply_large_frac_min), 1e-6),
                        np.maximum(float(args.v14b_apply_no_yaw_frac_max) - no_frac[apply], 0.0) / max(float(args.v14b_apply_no_yaw_frac_max), 1e-6),
                        np.maximum(sign_agree_frac[apply] - float(args.v14b_apply_sign_agree_min), 0.0) / max(1.0 - float(args.v14b_apply_sign_agree_min), 1e-6),
                    ]
                )
            ),
            0.25,
            1.0,
        )
    # Keep ambiguous rows available for curve diagnostics but out of supervised mode CE.
    if str(args.mode_semantics) in {"v14_observable_binary", "v14b_observable_summary_binary"} and bool(args.v14_abstain_ambiguous):
        mode_valid = (keep | apply) & supervised_profile_mask
    else:
        mode_valid = (label >= 0) & supervised_profile_mask

    has_no = curve_counts[:, 0] > 0
    has_neg = (curve_counts[:, 1] + curve_counts[:, 3]) > 0
    has_pos = (curve_counts[:, 2] + curve_counts[:, 4]) > 0
    complete_yaw_bank = yaw_needed & has_no & has_neg & has_pos
    candidate_incomplete = yaw_needed & ~complete_yaw_bank
    if args.require_complete_yaw_bank:
        mode_valid = mode_valid & complete_yaw_bank

    best_bin = np.argmax(curve_scores, axis=1).astype(np.int64)
    best_curve_yaw = curve_yaw[np.arange(ep.shape[0]), best_bin]
    direction = np.zeros((ep.shape[0],), dtype=np.int64)
    direction[best_curve_yaw < -1e-6] = -1
    direction[best_curve_yaw > 1e-6] = 1

    def _collect_source_confounded(keep_mask: np.ndarray, apply_mask: np.ndarray) -> list[int]:
        confounded = []
        for e in np.unique(ep[yaw_needed]).astype(int).tolist():
            em = ep == int(e)
            keep_m = em & keep_mask
            apply_m = em & apply_mask
            keep_dom, keep_frac = _dominant_profile(source_profile[keep_m])
            apply_dom, apply_frac = _dominant_profile(source_profile[apply_m])
            clean_same_profile = False
            for p in sorted(set(source_profile[em].astype(str).tolist())):
                pm = em & (source_profile.astype(str) == p)
                if np.any(pm & keep_mask) and np.any(pm & apply_mask):
                    clean_same_profile = True
                    break
            if (
                np.any(keep_m)
                and np.any(apply_m)
                and keep_dom != apply_dom
                and keep_frac >= float(args.source_confound_dominance)
                and apply_frac >= float(args.source_confound_dominance)
                and not clean_same_profile
            ):
                confounded.append(int(e))
        return confounded

    source_confounded_eps = _collect_source_confounded(keep, apply)
    if bool(args.drop_source_confounded_supervision) and source_confounded_eps:
        confounded_mask = np.isin(ep, np.asarray(source_confounded_eps, dtype=np.int64))
        keep = keep & ~confounded_mask
        apply = apply & ~confounded_mask
        if str(args.mode_semantics) in {"v14_observable_binary", "v14b_observable_summary_binary"}:
            if bool(args.v14_abstain_ambiguous):
                small_mode = np.zeros_like(valid, dtype=bool)
            else:
                small_mode = valid & supervised_profile_mask & ~(keep | apply)
            ambiguous = valid & supervised_profile_mask & ~(keep | apply)
            mode_valid = (keep | apply) & supervised_profile_mask if bool(args.v14_abstain_ambiguous) else (label >= 0) & supervised_profile_mask
        else:
            small_mode = valid & supervised_profile_mask & ~keep & ~apply
            mode_valid = (label >= 0) & supervised_profile_mask
        label = np.full((ep.shape[0],), -1, dtype=np.int64)
        confidence = np.zeros((ep.shape[0],), dtype=np.float32)
        label[keep] = 0
        label[small_mode] = 1
        label[apply] = 2
        confidence[keep] = np.clip((float(args.keep_margin) - yaw_advantage[keep]) / conf_scale, 0.05, 1.0)
        confidence[small_mode] = np.clip(np.abs(middle_adv[small_mode]) / conf_scale, 0.05, 0.5)
        confidence[apply] = np.clip(large_advantage[apply] / conf_scale, 0.05, 1.0)
        if str(args.mode_semantics) in {"v14_observable_binary", "v14b_observable_summary_binary"}:
            confidence[keep] *= np.clip(
                (
                    (float(args.v14_keep_proxy_yaw_abs_max) - proxy_dyaw_abs[keep]) / max(float(args.v14_keep_proxy_yaw_abs_max), 1e-6)
                    if str(args.mode_semantics) == "v14_observable_binary"
                    else np.maximum(
                        no_frac[keep] - float(args.v14b_keep_soft_no_yaw_frac_min),
                        (scope_count[keep] - float(args.v14b_apply_scope_size_max)) / max(float(args.v14b_keep_scope_size_min), 1.0),
                    )
                ),
                0.25,
                1.0,
            )
            confidence[apply] *= np.clip(
                (
                    (proxy_dyaw_abs[apply] - float(args.v14_apply_proxy_yaw_abs_min)) / max(0.05, float(args.v14_apply_proxy_yaw_abs_min))
                    if str(args.mode_semantics) == "v14_observable_binary"
                    else np.minimum.reduce(
                        [
                            np.maximum(large_frac[apply] - float(args.v14b_apply_large_frac_min), 0.0) / max(1.0 - float(args.v14b_apply_large_frac_min), 1e-6),
                            np.maximum(float(args.v14b_apply_no_yaw_frac_max) - no_frac[apply], 0.0) / max(float(args.v14b_apply_no_yaw_frac_max), 1e-6),
                            np.maximum(sign_agree_frac[apply] - float(args.v14b_apply_sign_agree_min), 0.0) / max(1.0 - float(args.v14b_apply_sign_agree_min), 1e-6),
                        ]
                    )
                ),
                0.25,
                1.0,
            )
        source_confounded_eps = _collect_source_confounded(keep, apply)

    out = dict(data)
    out["b2_yaw_cost_curve_scores_v11"] = curve_scores
    out["b2_yaw_cost_curve_best_index_v11"] = curve_idx
    out["b2_yaw_cost_curve_best_yaw_v11"] = curve_yaw
    out["b2_yaw_cost_curve_counts_v11"] = curve_counts
    out["yaw_advantage_cont_v11"] = yaw_advantage.astype(np.float32)
    out["yaw_small_advantage_v11"] = np.where(np.isfinite(small_advantage), small_advantage, 0.0).astype(np.float32)
    out["yaw_large_advantage_v11"] = np.where(np.isfinite(large_advantage), large_advantage, 0.0).astype(np.float32)
    out["yaw_mode3_label_v11"] = label.astype(np.int64)
    out["yaw_mode_valid_v11"] = mode_valid.astype(np.float32)
    out["yaw_mode_confidence_v11"] = confidence.astype(np.float32)
    out["yaw_keep_v11"] = keep.astype(np.float32)
    out["yaw_small_v11"] = small_mode.astype(np.float32)
    out["yaw_apply_v11"] = apply.astype(np.float32)
    out["yaw_mode_ambiguous_v11"] = ambiguous.astype(np.float32)
    out["yaw_direction_label_v11"] = direction.astype(np.int64)
    out["yaw_candidate_bank_complete_v12"] = complete_yaw_bank.astype(np.float32)
    out["yaw_candidate_bank_incomplete_v12"] = candidate_incomplete.astype(np.float32)
    out["source_profile_v12"] = source_profile.astype("U32")
    out["yaw_mode_supervised_profile_v13"] = supervised_profile_mask.astype(np.float32)
    out["yaw_mode_diagnostic_profile_v13"] = diagnostic_profile_mask.astype(np.float32)
    out["yaw_mode_semantics_v14"] = np.full(
        (ep.shape[0],),
        2 if str(args.mode_semantics) == "v14b_observable_summary_binary" else (1 if str(args.mode_semantics) == "v14_observable_binary" else 0),
        dtype=np.int64,
    )
    out["yaw_mode_proxy_dyaw_abs_v14"] = proxy_dyaw_abs.astype(np.float32)
    out["yaw_mode_observable_keep_v14"] = (
        (
            valid & supervised_profile_mask & (proxy_dyaw_abs <= float(args.v14_keep_proxy_yaw_abs_max))
            if str(args.mode_semantics) != "v14b_observable_summary_binary"
            else valid & supervised_profile_mask & (
                (no_frac >= float(args.v14b_keep_no_yaw_frac_min))
                | (scope_count >= float(args.v14b_keep_scope_size_min))
                | (
                    (no_frac >= float(args.v14b_keep_soft_no_yaw_frac_min))
                    & (yaw_abs_mean <= float(args.v14b_keep_abs_mean_max))
                    & (yaw_std <= float(args.v14b_keep_std_max))
                )
                | (
                    (yaw_abs_mean <= float(args.v14b_keep_tight_abs_mean_max))
                    & (yaw_std <= float(args.v14b_keep_tight_std_max))
                )
            )
        )
    ).astype(np.float32)
    out["yaw_mode_observable_apply_v14"] = (
        (
            valid & supervised_profile_mask & (proxy_dyaw_abs >= float(args.v14_apply_proxy_yaw_abs_min))
            if str(args.mode_semantics) != "v14b_observable_summary_binary"
            else valid & supervised_profile_mask & (
                (no_frac <= float(args.v14b_apply_no_yaw_frac_max))
                & (large_frac >= float(args.v14b_apply_large_frac_min))
                & (scope_count <= float(args.v14b_apply_scope_size_max))
                & (yaw_abs_mean >= float(args.v14b_apply_abs_mean_min))
                & (yaw_std >= float(args.v14b_apply_std_min))
                & (sign_agree_frac >= float(args.v14b_apply_sign_agree_min))
            )
        )
    ).astype(np.float32)
    out["yaw_mode_no_yaw_frac_v14b"] = no_frac.astype(np.float32)
    out["yaw_mode_small_frac_v14b"] = small_frac.astype(np.float32)
    out["yaw_mode_large_frac_v14b"] = large_frac.astype(np.float32)
    out["yaw_mode_pos_frac_v14b"] = pos_frac.astype(np.float32)
    out["yaw_mode_neg_frac_v14b"] = neg_frac.astype(np.float32)
    out["yaw_mode_sign_agree_frac_v14b"] = sign_agree_frac.astype(np.float32)
    out["yaw_mode_scope_size_v14b"] = scope_count.astype(np.float32)
    out["yaw_mode_yaw_abs_mean_v14b"] = yaw_abs_mean.astype(np.float32)
    out["yaw_mode_yaw_std_v14b"] = yaw_std.astype(np.float32)

    split_info = _pick_split(
        ep,
        keep,
        small_mode,
        apply,
        int(args.seed),
        float(args.val_ratio),
        int(args.min_train_yaw_apply_eps),
        int(args.min_train_yaw_keep_eps),
    )
    out["split_v11"] = split_info.pop("split_v11")

    output_npz = Path(args.output_npz)
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    if args.compressed:
        np.savez_compressed(output_npz, **out)
    else:
        # This dataset carries image/depth arrays from the source candidate bank.
        # Plain NPZ is faster; use --compressed when disk is tighter than time.
        np.savez(output_npz, **out)

    masks = {
        "teacher_ready": np.asarray(data.get("teacher_ready_v1", np.zeros_like(yaw_needed, dtype=np.float32))) > 0.5,
        "xy_block": np.asarray(data.get("xy_block_v1", np.zeros_like(yaw_needed, dtype=np.float32))) > 0.5,
        "yaw_needed": yaw_needed,
        "far_negative": np.asarray(data.get("far_negative_v1", np.zeros_like(yaw_needed, dtype=np.float32))) > 0.5,
        "yaw_keep": keep,
        "yaw_small": small_mode,
        "yaw_apply": apply,
        "yaw_ambiguous": ambiguous,
        "yaw_mode_valid": mode_valid,
    }
    bucket_counts = {name: _episode_counts(ep, m) for name, m in masks.items()}
    source_counts = {str(k): int(v) for k, v in zip(*np.unique(source, return_counts=True))}
    source_bucket_counts = {}
    for name, m in masks.items():
        source_bucket_counts[name] = {str(k): int(v) for k, v in zip(*np.unique(source[m], return_counts=True))} if np.any(m) else {}

    profile_mode_counts = {}
    for p in sorted(set(source_profile.astype(str).tolist())):
        pm = source_profile.astype(str) == p
        profile_mode_counts[p] = {
            "rows": int(np.sum(pm)),
            "diagnostic_rows": int(np.sum(pm & diagnostic_profile_mask)),
            "supervised_rows": int(np.sum(pm & supervised_profile_mask)),
            "yaw_needed_rows": int(np.sum(pm & yaw_needed)),
            "keep_rows": int(np.sum(pm & keep)),
            "keep_eps": int(np.unique(ep[pm & keep]).size),
            "small_rows": int(np.sum(pm & small_mode)),
            "small_eps": int(np.unique(ep[pm & small_mode]).size),
            "apply_rows": int(np.sum(pm & apply)),
            "apply_eps": int(np.unique(ep[pm & apply]).size),
        }
    max_profile_apply_eps = max((v["apply_eps"] for v in profile_mode_counts.values()), default=0)
    max_profile_keep_eps = max((v["keep_eps"] for v in profile_mode_counts.values()), default=0)

    insufficient = []
    if bucket_counts["yaw_apply"]["eps"] < int(args.min_yaw_apply_eps):
        insufficient.append(f"yaw_apply_eps {bucket_counts['yaw_apply']['eps']} < {args.min_yaw_apply_eps}")
    if bucket_counts["yaw_keep"]["eps"] < int(args.min_yaw_keep_eps):
        insufficient.append(f"yaw_keep_eps {bucket_counts['yaw_keep']['eps']} < {args.min_yaw_keep_eps}")
    if split_info["val_yaw_apply_eps"] < int(args.min_val_yaw_apply_eps):
        insufficient.append(f"val_yaw_apply_eps {split_info['val_yaw_apply_eps']} < {args.min_val_yaw_apply_eps}")
    if split_info["val_yaw_keep_eps"] < int(args.min_val_yaw_keep_eps):
        insufficient.append(f"val_yaw_keep_eps {split_info['val_yaw_keep_eps']} < {args.min_val_yaw_keep_eps}")
    if split_info["train_yaw_apply_eps"] < int(args.min_train_yaw_apply_eps):
        insufficient.append(f"train_yaw_apply_eps {split_info['train_yaw_apply_eps']} < {args.min_train_yaw_apply_eps}")
    if split_info["train_yaw_keep_eps"] < int(args.min_train_yaw_keep_eps):
        insufficient.append(f"train_yaw_keep_eps {split_info['train_yaw_keep_eps']} < {args.min_train_yaw_keep_eps}")
    if args.require_complete_yaw_bank and np.any(candidate_incomplete):
        insufficient.append(f"candidate_incomplete_rows {int(np.sum(candidate_incomplete))} > 0")
    if source_confounded_eps:
        insufficient.append(f"source_confounded_eps {source_confounded_eps}")
    if int(args.min_profile_yaw_apply_eps) > 0 and max_profile_apply_eps < int(args.min_profile_yaw_apply_eps):
        insufficient.append(
            f"profile_yaw_apply_eps {max_profile_apply_eps} < {args.min_profile_yaw_apply_eps}"
        )
    if int(args.min_profile_yaw_keep_eps) > 0 and max_profile_keep_eps < int(args.min_profile_yaw_keep_eps):
        insufficient.append(
            f"profile_yaw_keep_eps {max_profile_keep_eps} < {args.min_profile_yaw_keep_eps}"
        )

    candidate_bank_complete = {
        "yaw_needed_rows": int(np.sum(yaw_needed)),
        "complete_rows": int(np.sum(complete_yaw_bank)),
        "complete_rate": float(np.mean(complete_yaw_bank[yaw_needed])) if np.any(yaw_needed) else 0.0,
        "incomplete_rows": int(np.sum(candidate_incomplete)),
    }

    manifest = {
        "schema": str(args.schema_name),
        "input_npz": str(args.input_npz),
        "output_npz": str(args.output_npz),
        "bin_names": BIN_NAMES,
        "thresholds": vars(args),
        "rows": int(ep.shape[0]),
        "episodes": int(np.unique(ep).size),
        "source_counts": source_counts,
        "source_bucket_counts": source_bucket_counts,
        "source_profile_counts": {str(k): int(v) for k, v in zip(*np.unique(source_profile, return_counts=True))},
        "source_profile_policy": {
            "supervised_profiles": sorted(supervised_profiles) if supervised_profiles else "all",
            "diagnostic_profiles": sorted(diagnostic_profiles),
            "profile_mode_counts": profile_mode_counts,
            "max_profile_yaw_apply_eps": int(max_profile_apply_eps),
            "max_profile_yaw_keep_eps": int(max_profile_keep_eps),
        },
        "source_confounded_eps": source_confounded_eps,
        "candidate_bank_complete": candidate_bank_complete,
        "bucket_counts": bucket_counts,
        "split": split_info,
        "passes_gate": not insufficient,
        "insufficient_reasons": insufficient,
        "runtime_provider_frozen": True,
        "offline_only": True,
    }
    Path(args.manifest_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.manifest_json).write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    gate = {
        "passes_gate": not insufficient,
        "insufficient_reasons": insufficient,
        "dataset_npz": str(args.output_npz),
        "bucket_counts": bucket_counts,
        "split": split_info,
        "source_confounded_eps": source_confounded_eps,
        "source_profile_policy": manifest["source_profile_policy"],
        "candidate_bank_complete": candidate_bank_complete,
        "decision": "offline_only_ok" if not insufficient else "diagnostic_only",
    }
    Path(args.gate_report_json).write_text(json.dumps(gate, indent=2, ensure_ascii=False))

    focus = [int(x) for x in args.focus_episodes.split(",") if x.strip()]
    episode_audit = {}
    for e in sorted(set(focus + np.unique(ep[mode_valid]).astype(int).tolist())):
        em = ep == int(e)
        if not np.any(em):
            continue
        episode_audit[str(e)] = {
            "rows": int(np.sum(em)),
            "yaw_needed_rows": int(np.sum(em & yaw_needed)),
            "keep_rows": int(np.sum(em & keep)),
            "small_rows": int(np.sum(em & small_mode)),
            "apply_rows": int(np.sum(em & apply)),
            "ambiguous_rows": int(np.sum(em & ambiguous)),
            "yaw_advantage_mean": _json_float(np.mean(yaw_advantage[em & yaw_needed])) if np.any(em & yaw_needed) else None,
            "yaw_advantage_p10_p50_p90": [
                _json_float(x) for x in np.percentile(yaw_advantage[em & yaw_needed], [10, 50, 90])
            ] if np.any(em & yaw_needed) else [],
            "no_yaw_score_mean": _json_float(np.mean(no_score[em & no_valid])) if np.any(em & no_valid) else None,
            "small_score_mean": _json_float(np.mean(small_score[em & small_valid])) if np.any(em & small_valid) else None,
            "large_score_mean": _json_float(np.mean(large_score[em & large_valid])) if np.any(em & large_valid) else None,
            "mode_confidence_mean": _json_float(np.mean(confidence[em & mode_valid])) if np.any(em & mode_valid) else None,
            "source_counts": {str(k): int(v) for k, v in zip(*np.unique(source[em], return_counts=True))},
            "source_profile_counts": {str(k): int(v) for k, v in zip(*np.unique(source_profile[em], return_counts=True))},
            "candidate_bank_complete_rows": int(np.sum(em & complete_yaw_bank)),
            "candidate_bank_complete_rate": float(np.mean(complete_yaw_bank[em & yaw_needed])) if np.any(em & yaw_needed) else 0.0,
            "bin_available_rows": {
                name: int(np.sum(em & yaw_needed & (curve_counts[:, i] > 0))) for i, name in enumerate(BIN_NAMES)
            },
        }
    Path(args.episode_audit_json).write_text(json.dumps(episode_audit, indent=2, ensure_ascii=False))
    print(json.dumps(gate, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
