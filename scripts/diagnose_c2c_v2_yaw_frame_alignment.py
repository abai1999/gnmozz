#!/usr/bin/env python3
"""Diagnose C2C v2 yaw/frame alignment on entry-feasible but control-blocked rows.

This is an offline-only diagnostic.  It uses privileged frame residual labels to
ask why rows with feasible entry yaw are still blocked by the non-privileged yaw
control gate.  It does not change runtime policy or gate thresholds.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _wrap_yaw_to_symmetry(yaw: float, period: float) -> float:
    if not (np.isfinite(yaw) and np.isfinite(period) and period > 0.0):
        return float("nan")
    half = 0.5 * float(period)
    return float(((float(yaw) + half) % float(period)) - half)


def _pose_yaw_delta_from_poses(ref: np.ndarray, tgt: np.ndarray) -> float:
    ref = np.asarray(ref, dtype=np.float64).reshape(7)
    tgt = np.asarray(tgt, dtype=np.float64).reshape(7)
    if not (np.all(np.isfinite(ref)) and np.all(np.isfinite(tgt))):
        return float("nan")
    r_ref = Rotation.from_quat(ref[3:7])
    r_tgt = Rotation.from_quat(tgt[3:7])
    rel = (r_tgt * r_ref.inv()).as_rotvec()
    return float(rel[2])


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _finite(value: Any) -> bool:
    return bool(np.isfinite(_safe_float(value)))


def _corr(a: Iterable[float], b: Iterable[float]) -> float:
    aa = np.asarray(list(a), dtype=np.float64)
    bb = np.asarray(list(b), dtype=np.float64)
    mask = np.isfinite(aa) & np.isfinite(bb)
    if int(np.count_nonzero(mask)) < 2:
        return 0.0
    aa = aa[mask]
    bb = bb[mask]
    if float(np.std(aa)) <= 1.0e-12 or float(np.std(bb)) <= 1.0e-12:
        return 0.0
    return float(np.corrcoef(aa, bb)[0, 1])


def _sign_match(proxy: float, target: float, *, eps: float = 1.0e-6) -> bool | None:
    if not (np.isfinite(proxy) and np.isfinite(target)):
        return None
    if abs(float(proxy)) <= eps or abs(float(target)) <= eps:
        return None
    return bool(np.sign(proxy) == np.sign(target))


def _mean_bool(values: list[bool | None]) -> float:
    clean = [bool(v) for v in values if v is not None]
    return float(np.mean(clean)) if clean else 0.0


def _proxy_mapping(row: Mapping[str, Any]) -> Mapping[str, Any]:
    proxy = row.get("proxy_local_geometry_error") if isinstance(row.get("proxy_local_geometry_error"), Mapping) else {}
    return proxy


def _estimated_mapping(row: Mapping[str, Any]) -> Mapping[str, Any]:
    est = row.get("estimated_basin_error") if isinstance(row.get("estimated_basin_error"), Mapping) else {}
    return est


def _proxy_residual_yaw(row: Mapping[str, Any]) -> float:
    proxy = _proxy_mapping(row)
    est = _estimated_mapping(row)
    return _safe_float(proxy.get("dyaw", est.get("dyaw", 0.0)), 0.0)


def _image_axis_yaw(row: Mapping[str, Any]) -> float:
    proxy = _proxy_mapping(row)
    if "image_axis_yaw" in proxy:
        return _safe_float(proxy.get("image_axis_yaw"), 0.0)
    # Backwards compatibility for old relabels where PCA image-axis yaw was
    # serialized as dyaw.  New relabels keep dyaw as residual yaw and put the
    # image/PCA diagnostic in image_axis_yaw.
    return _proxy_residual_yaw(row)


def _proxy_yaw(row: Mapping[str, Any]) -> float:
    return _image_axis_yaw(row)


def _proxy_valid(row: Mapping[str, Any]) -> bool:
    proxy = _proxy_mapping(row)
    est = _estimated_mapping(row)
    return bool(proxy.get("valid", False) or est.get("yaw_valid", False))


def _privileged_yaw(row: Mapping[str, Any]) -> float:
    nested = row.get("true_basin_error_t") if isinstance(row.get("true_basin_error_t"), Mapping) else {}
    return _safe_float(nested.get("dyaw", row.get("privileged_dyaw", float("nan"))), float("nan"))


def _xy_error(row: Mapping[str, Any]) -> float:
    if "xy_error" in row:
        return _safe_float(row.get("xy_error"), float("nan"))
    nested = row.get("true_basin_error_t") if isinstance(row.get("true_basin_error_t"), Mapping) else {}
    dx = _safe_float(nested.get("dx", row.get("privileged_dx", float("nan"))), float("nan"))
    dy = _safe_float(nested.get("dy", row.get("privileged_dy", float("nan"))), float("nan"))
    return float(math.hypot(dx, dy)) if np.isfinite(dx) and np.isfinite(dy) else float("nan")


def _pose_yaw_delta(row: Mapping[str, Any]) -> float:
    ref = row.get("reference_frame_pose_7d")
    tgt = row.get("target_frame_pose_7d")
    if ref is None or tgt is None:
        return float("nan")
    try:
        return float(_pose_yaw_delta_from_poses(ref, tgt))
    except Exception:
        return float("nan")


def _symmetry_period(row: Mapping[str, Any], default: float) -> float:
    contract = row.get("frame_contract") if isinstance(row.get("frame_contract"), Mapping) else {}
    yaw_mode = str(contract.get("yaw_mode", row.get("yaw_mode", "")))
    if "square" in yaw_mode:
        return float(np.pi / 2.0)
    return float(default)


def _symmetry_candidates(raw_yaw: float, period: float) -> list[tuple[int, float]]:
    if not (np.isfinite(raw_yaw) and np.isfinite(period) and period > 0.0):
        return [(0, float(raw_yaw))]
    candidates: list[tuple[int, float]] = []
    for k in range(-4, 5):
        yaw = float(raw_yaw - float(k) * float(period))
        yaw = float(((yaw + np.pi) % (2.0 * np.pi)) - np.pi)
        candidates.append((k, yaw))
    # De-duplicate candidates that wrap to the same angle.
    out: list[tuple[int, float]] = []
    seen: set[int] = set()
    for k, yaw in candidates:
        key = int(round(yaw * 1.0e6))
        if key not in seen:
            seen.add(key)
            out.append((k, yaw))
    return out


def _best_symmetry_alias(proxy_yaw: float, raw_yaw: float, period: float) -> tuple[int, float, float]:
    candidates = _symmetry_candidates(raw_yaw, period)
    if not candidates or not np.isfinite(proxy_yaw):
        return 0, float("nan"), float("inf")
    k, yaw = min(candidates, key=lambda item: abs(float(proxy_yaw) - float(item[1])))
    return int(k), float(yaw), float(abs(float(proxy_yaw) - float(yaw)))


def _symmetry_aware_proxy_yaw(proxy_yaw: float, period: float) -> float:
    if not (np.isfinite(proxy_yaw) and np.isfinite(period) and period > 0.0):
        return float("nan")
    return float(-_wrap_yaw_to_symmetry(proxy_yaw, period))


def _finite_pair_mask(proxy: Iterable[float], priv: Iterable[float]) -> np.ndarray:
    proxy_arr = np.asarray(list(proxy), dtype=np.float64)
    priv_arr = np.asarray(list(priv), dtype=np.float64)
    return np.isfinite(proxy_arr) & np.isfinite(priv_arr)


def _baseline_summary(proxy: Iterable[float], priv: Iterable[float]) -> dict[str, float]:
    proxy_arr = np.asarray(list(proxy), dtype=np.float64)
    priv_arr = np.asarray(list(priv), dtype=np.float64)
    mask = np.isfinite(proxy_arr) & np.isfinite(priv_arr)
    if not np.any(mask):
        return {
            "mae": 0.0,
            "bias": 0.0,
            "bias_corrected_mae": 0.0,
            "corr": 0.0,
            "sign_match_rate": 0.0,
            "bias_corrected_corr": 0.0,
            "bias_corrected_sign_match_rate": 0.0,
            "residual_std": 0.0,
        }
    proxy_arr = proxy_arr[mask]
    priv_arr = priv_arr[mask]
    residual = proxy_arr - priv_arr
    bias = float(np.mean(residual)) if residual.size else 0.0
    corrected = proxy_arr - bias
    corrected_residual = corrected - priv_arr
    return {
        "mae": float(np.mean(np.abs(residual))) if residual.size else 0.0,
        "bias": bias,
        "bias_corrected_mae": float(np.mean(np.abs(corrected_residual))) if corrected_residual.size else 0.0,
        "corr": _corr(proxy_arr, priv_arr),
        "sign_match_rate": _mean_bool([_sign_match(p, t) for p, t in zip(proxy_arr, priv_arr)]),
        "bias_corrected_corr": _corr(corrected, priv_arr),
        "bias_corrected_sign_match_rate": _mean_bool([_sign_match(p, t) for p, t in zip(corrected, priv_arr)]),
        "residual_std": float(np.std(residual)) if residual.size else 0.0,
    }


def _diagnosis_label(row: Mapping[str, Any], item: Mapping[str, Any]) -> str:
    proxy = _safe_float(item.get("proxy_yaw"))
    priv = _safe_float(item.get("privileged_yaw"))
    raw = _safe_float(item.get("raw_pose_dyaw"))
    best_alias = _safe_float(item.get("best_symmetry_alias_yaw"))
    best_alias_k = int(item.get("best_symmetry_alias_k", 0))
    proxy_conf = _safe_float(row.get("yaw_observability_frame_confidence", row.get("source_frame_confidence", 0.0)), 0.0)
    proxy_obs = _safe_float(row.get("yaw_observability_frame_observability", row.get("source_frame_observability", 0.0)), 0.0)

    if not _proxy_valid(row) or (abs(proxy) <= 1.0e-6 and proxy_conf <= 1.0e-6):
        return "no_proxy_signal"
    if str(item.get("proxy_yaw_semantics", "")) == "image_pca_axis_yaw" and abs(proxy - priv) > 0.20:
        return "image_axis_not_jaw_local_residual"
    if str(row.get("yaw_observability_primary_blocker", "")) == "wrist_occluded":
        return "occlusion_blocks_frame_axis"
    if np.isfinite(proxy) and np.isfinite(priv) and abs(proxy + priv) < abs(proxy - priv) and abs(proxy) > 1.0e-6 and abs(priv) > 1.0e-6:
        return "sign_flip_candidate"
    if np.isfinite(best_alias) and np.isfinite(priv) and best_alias_k != 0 and abs(proxy - best_alias) + 1.0e-9 < abs(proxy - priv):
        return "symmetry_alias_candidate"
    if np.isfinite(raw) and np.isfinite(priv) and abs(_wrap_yaw_to_symmetry(raw, np.pi / 2.0) - priv) > 0.02:
        return "symmetry_wrapping_mismatch"
    if proxy_obs < 0.10:
        return "weak_frame_observability"
    return "frame_definition_drift_candidate"


def _enrich(row: Mapping[str, Any], *, default_symmetry_period: float) -> dict[str, Any]:
    proxy = _proxy_yaw(row)
    proxy_residual = _proxy_residual_yaw(row)
    image_axis = _image_axis_yaw(row)
    priv = _privileged_yaw(row)
    raw = _pose_yaw_delta(row)
    period = _symmetry_period(row, default_symmetry_period)
    wrapped_from_pose = _wrap_yaw_to_symmetry(raw, period) if np.isfinite(raw) else float("nan")
    best_k, best_alias, best_alias_err = _best_symmetry_alias(proxy, raw, period)
    item = {
        "episode_idx": int(row.get("episode_idx", -1)),
        "step_idx": int(row.get("step_idx", row.get("step", -1))),
        "stage_name": str(row.get("stage_name", "")),
        "skill_name": str(row.get("skill_name", "")),
        "failure_bucket": str(row.get("failure_bucket", "")),
        "visual_observability_class": str(row.get("visual_observability_class", "")),
        "yaw_observability_class": str(row.get("yaw_observability_class", "")),
        "yaw_observability_primary_blocker": str(row.get("yaw_observability_primary_blocker", "")),
        "yaw_observability_blocker_combo": str(row.get("yaw_observability_blocker_combo", "")),
        "frame_confidence": _safe_float(row.get("yaw_observability_frame_confidence", row.get("source_frame_confidence", 0.0)), 0.0),
        "frame_observability": _safe_float(row.get("yaw_observability_frame_observability", row.get("source_frame_observability", 0.0)), 0.0),
        "frame_axis_strength": _safe_float(row.get("yaw_observability_frame_axis_strength", row.get("source_frame_axis_strength", 0.0)), 0.0),
        "wrist_occluded": bool(row.get("yaw_observability_wrist_occluded", row.get("wrist_is_occluded", False))),
        "xy_error": _xy_error(row),
        "near_basin_shell": bool(row.get("near_basin_shell", False)),
        "micro_entry_ready": bool(row.get("micro_entry_ready", False)),
        "proxy_yaw": float(proxy),
        "proxy_yaw_semantics": "image_pca_axis_yaw" if "image_axis_yaw" in _proxy_mapping(row) else "legacy_proxy_dyaw",
        "symmetry_aware_proxy_yaw": float(_symmetry_aware_proxy_yaw(proxy, period)),
        "proxy_residual_yaw": float(proxy_residual),
        "image_axis_yaw": float(image_axis),
        "privileged_yaw": float(priv),
        "raw_pose_dyaw": float(raw),
        "symmetry_period": float(period),
        "wrapped_pose_dyaw": float(wrapped_from_pose),
        "pose_wrap_minus_privileged": float(wrapped_from_pose - priv) if np.isfinite(wrapped_from_pose) and np.isfinite(priv) else float("nan"),
        "best_symmetry_alias_k": int(best_k),
        "best_symmetry_alias_yaw": float(best_alias),
        "best_symmetry_alias_abs_error": float(best_alias_err),
        "proxy_minus_privileged": float(proxy - priv) if np.isfinite(proxy) and np.isfinite(priv) else float("nan"),
        "proxy_plus_privileged": float(proxy + priv) if np.isfinite(proxy) and np.isfinite(priv) else float("nan"),
        "proxy_privileged_sign_match": _sign_match(proxy, priv),
        "proxy_neg_privileged_abs_error": float(abs(proxy + priv)) if np.isfinite(proxy) and np.isfinite(priv) else float("nan"),
        "proxy_privileged_abs_error": float(abs(proxy - priv)) if np.isfinite(proxy) and np.isfinite(priv) else float("nan"),
        "symmetry_aware_proxy_minus_privileged": float(_symmetry_aware_proxy_yaw(proxy, period) - priv) if np.isfinite(proxy) and np.isfinite(priv) else float("nan"),
        "residual_yaw_privileged_abs_error": float(abs(proxy_residual - priv)) if np.isfinite(proxy_residual) and np.isfinite(priv) else float("nan"),
        "proxy_valid": bool(_proxy_valid(row)),
        "proxy_yaw_valid": bool(_proxy_mapping(row).get("yaw_valid", True)),
        "proxy_yaw_reason": str(_proxy_mapping(row).get("yaw_reason", "")),
        "proxy_reason": str(_proxy_mapping(row).get("reason", "")),
    }
    item["diagnosis_label"] = _diagnosis_label(row, item)
    return item


def _group_counts(items: list[dict[str, Any]], key: str) -> dict[str, int]:
    return dict(Counter(str(item.get(key, "")) for item in items))


def _group_metric(items: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        groups[str(item.get(key, ""))].append(item)
    out: list[dict[str, Any]] = []
    for value, subset in sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        proxy = [_safe_float(r.get("proxy_yaw")) for r in subset]
        priv = [_safe_float(r.get("privileged_yaw")) for r in subset]
        symm = [_safe_float(r.get("symmetry_aware_proxy_yaw")) for r in subset]
        symm_summary = _baseline_summary(symm, priv)
        out.append(
            {
                key: value,
                "num_rows": int(len(subset)),
                "near_basin_shell_rows": int(sum(bool(r.get("near_basin_shell", False)) for r in subset)),
                "micro_entry_ready_rows": int(sum(bool(r.get("micro_entry_ready", False)) for r in subset)),
                "sign_match_rate": _mean_bool([r.get("proxy_privileged_sign_match") for r in subset]),
                "proxy_privileged_corr": _corr(proxy, priv),
                "proxy_privileged_mae": float(np.nanmean([_safe_float(r.get("proxy_privileged_abs_error")) for r in subset])) if subset else 0.0,
                "symmetry_aware_proxy_mae": float(symm_summary["mae"]),
                "symmetry_aware_proxy_bias": float(symm_summary["bias"]),
                "symmetry_aware_proxy_bias_corrected_mae": float(symm_summary["bias_corrected_mae"]),
                "symmetry_aware_proxy_bias_corrected_corr": float(symm_summary["bias_corrected_corr"]),
                "diagnosis_counts": _group_counts(subset, "diagnosis_label"),
            }
        )
    return out


def _write_plots(items: list[dict[str, Any]], output_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    if not items:
        return
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    proxy = np.asarray([_safe_float(r.get("proxy_yaw")) for r in items], dtype=np.float64)
    symm = np.asarray([_safe_float(r.get("symmetry_aware_proxy_yaw")) for r in items], dtype=np.float64)
    priv = np.asarray([_safe_float(r.get("privileged_yaw")) for r in items], dtype=np.float64)
    obs = np.asarray([_safe_float(r.get("frame_observability")) for r in items], dtype=np.float64)
    xy = np.asarray([_safe_float(r.get("xy_error")) for r in items], dtype=np.float64)
    labels = [str(r.get("diagnosis_label", "")) for r in items]
    label_names = sorted(set(labels))
    color_map = {name: idx for idx, name in enumerate(label_names)}
    colors = np.asarray([color_map[name] for name in labels], dtype=np.float64)

    fig, ax = plt.subplots(figsize=(8, 7))
    sc = ax.scatter(priv, symm, c=colors, s=10, alpha=0.55, cmap="tab10")
    lim = max(0.12, float(np.nanmax(np.abs(np.concatenate([symm, priv])))) if symm.size else 0.12)
    ax.plot([-lim, lim], [-lim, lim], color="black", linewidth=1.0, label="y=x")
    ax.plot([-lim, lim], [lim, -lim], color="gray", linewidth=1.0, linestyle="--", label="sign flip")
    ax.axhline(0.0, color="black", linewidth=0.5)
    ax.axvline(0.0, color="black", linewidth=0.5)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("privileged dyaw")
    ax.set_ylabel("symmetry-aware proxy dyaw")
    ax.set_title("Symmetry-aware yaw proxy vs privileged yaw")
    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=sc.cmap(sc.norm(color_map[name])), markersize=6, label=name)
        for name in label_names
    ]
    ax.legend(handles=handles, loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(plot_dir / "symmetry_aware_proxy_vs_privileged_yaw.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(obs, np.abs(symm - priv), c=xy, s=10, alpha=0.55, cmap="viridis")
    ax.axvline(0.10, color="red", linestyle="--", linewidth=1.0, label="current obs gate")
    ax.set_xlabel("frame observability")
    ax.set_ylabel("|symmetry-aware proxy - privileged yaw|")
    ax.set_title("Symmetry-aware yaw error vs frame observability")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(plot_dir / "symmetry_aware_yaw_error_vs_frame_observability.png", dpi=160)
    plt.close(fig)

    if np.any(np.isfinite(symm)) and np.any(np.isfinite(priv)):
        mask = np.isfinite(symm) & np.isfinite(priv)
        bias = float(np.mean(symm[mask] - priv[mask])) if np.any(mask) else 0.0
        corrected = symm - bias
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist((corrected - priv)[mask], bins=min(20, max(5, int(np.count_nonzero(mask) / 2))), alpha=0.8, color="#4477aa")
        ax.axvline(0.0, color="black", linewidth=1.0)
        ax.axvline(bias, color="red", linestyle="--", linewidth=1.0, label=f"bias={bias:.4f}")
        ax.set_xlabel("bias-corrected residual yaw error")
        ax.set_ylabel("rows")
        ax.set_title("Residual after symmetry-aware bias correction")
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(plot_dir / "symmetry_aware_bias_corrected_residual_hist.png", dpi=160)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    counts = Counter(labels)
    names = [name for name, _ in counts.most_common()]
    vals = [counts[name] for name in names]
    ax.bar(names, vals)
    ax.set_ylabel("rows")
    ax.set_title("Diagnosis label counts")
    ax.tick_params(axis="x", labelrotation=35)
    fig.tight_layout()
    fig.savefig(plot_dir / "diagnosis_label_counts.png", dpi=160)
    plt.close(fig)

    top_eps = [int(ep) for ep, _ in Counter(int(r["episode_idx"]) for r in items).most_common(6)]
    fig, axes = plt.subplots(len(top_eps), 1, figsize=(12, max(3, 2.1 * len(top_eps))), sharex=False)
    if len(top_eps) == 1:
        axes = [axes]
    for ax, ep in zip(axes, top_eps):
        subset = sorted([r for r in items if int(r["episode_idx"]) == ep], key=lambda r: int(r["step_idx"]))
        steps = [int(r["step_idx"]) for r in subset]
        ax.plot(steps, [float(r["privileged_yaw"]) for r in subset], label="privileged", linewidth=1.5)
        ax.plot(steps, [float(r["proxy_yaw"]) for r in subset], label="raw image-axis proxy", linewidth=1.0, linestyle=":")
        ax.plot(steps, [float(r["symmetry_aware_proxy_yaw"]) for r in subset], label="symmetry-aware proxy", linewidth=1.2)
        ax.plot(
            steps,
            [float(r.get("symmetry_aware_proxy_bias_corrected", float("nan"))) for r in subset],
            label="bias-corrected proxy",
            linewidth=1.0,
            linestyle="--",
        )
        ax.plot(steps, [float(r["wrapped_pose_dyaw"]) for r in subset], label="wrapped pose", linewidth=1.0, linestyle="--")
        ax.axhline(0.0, color="black", linewidth=0.5)
        ax.set_ylabel(f"ep{ep:03d}")
        ax.legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("step")
    fig.suptitle("Yaw traces for top diagnostic episodes")
    fig.tight_layout()
    fig.savefig(plot_dir / "top_episode_yaw_traces.png", dpi=160)
    plt.close(fig)


def diagnose(
    rows: list[dict[str, Any]],
    *,
    stage_name: str,
    skill_type: str,
    visual_only: bool,
    near_basin_only: bool = False,
    min_frame_observability: float | None = None,
    require_not_wrist_occluded: bool = False,
    default_symmetry_period: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for row in rows:
        if stage_name and str(row.get("stage_name", "")) != stage_name:
            continue
        if skill_type and str(row.get("skill_type", "")) != skill_type:
            continue
        if visual_only and str(row.get("visual_observability_class", "")) != "visual_observable":
            continue
        if near_basin_only and not bool(row.get("near_basin_shell", False)):
            continue
        if min_frame_observability is not None:
            frame_observability = _safe_float(
                row.get("yaw_observability_frame_observability", row.get("source_frame_observability", 0.0)),
                0.0,
            )
            if frame_observability < float(min_frame_observability) - 1.0e-12:
                continue
        if require_not_wrist_occluded and bool(row.get("yaw_observability_wrist_occluded", row.get("wrist_is_occluded", False))):
            continue
        if not bool(row.get("yaw_entry_feasible", False)):
            continue
        if bool(row.get("yaw_control_observable", row.get("yaw_observable", False))):
            continue
        selected.append(_enrich(row, default_symmetry_period=default_symmetry_period))

    proxy = [_safe_float(r.get("proxy_yaw")) for r in selected]
    priv = [_safe_float(r.get("privileged_yaw")) for r in selected]
    symm = [_safe_float(r.get("symmetry_aware_proxy_yaw")) for r in selected]
    raw = [_safe_float(r.get("raw_pose_dyaw")) for r in selected]
    wrapped = [_safe_float(r.get("wrapped_pose_dyaw")) for r in selected]
    sign_matches = [r.get("proxy_privileged_sign_match") for r in selected]
    symm_summary = _baseline_summary(symm, priv)
    symm_bias = float(symm_summary["bias"])
    symm_corrected = [float(v - symm_bias) if np.isfinite(v) else float("nan") for v in symm]
    symm_corrected_summary = _baseline_summary(symm_corrected, priv)
    for item, corrected in zip(selected, symm_corrected):
        item["symmetry_aware_proxy_bias"] = symm_bias
        item["symmetry_aware_proxy_bias_corrected"] = float(corrected)
        item["symmetry_aware_proxy_bias_corrected_minus_privileged"] = float(corrected - _safe_float(item.get("privileged_yaw"))) if np.isfinite(corrected) and np.isfinite(_safe_float(item.get("privileged_yaw"))) else float("nan")
    report = {
        "schema_version": "yaw_frame_alignment_diagnostic_v1",
        "selection": {
            "stage_name": stage_name,
            "skill_type": skill_type,
            "visual_only": bool(visual_only),
            "near_basin_only": bool(near_basin_only),
            "min_frame_observability": None if min_frame_observability is None else float(min_frame_observability),
            "require_not_wrist_occluded": bool(require_not_wrist_occluded),
            "predicate": "yaw_entry_feasible && !yaw_control_observable",
            "baseline": "symmetry_aware_neg_wrap_proxy",
            "baseline_symmetry_period": float(default_symmetry_period),
        },
        "overall": {
            "num_rows": int(len(selected)),
            "episodes": int(len({int(r.get("episode_idx", -1)) for r in selected})),
            "near_basin_shell_rows": int(sum(bool(r.get("near_basin_shell", False)) for r in selected)),
            "micro_entry_ready_rows": int(sum(bool(r.get("micro_entry_ready", False)) for r in selected)),
            "proxy_valid_rows": int(sum(bool(r.get("proxy_valid", False)) for r in selected)),
            "proxy_privileged_sign_match_rate": _mean_bool(sign_matches),
            "proxy_neg_privileged_sign_match_rate": _mean_bool([
                _sign_match(_safe_float(r.get("proxy_yaw")), -_safe_float(r.get("privileged_yaw"))) for r in selected
            ]),
            "proxy_privileged_corr": _corr(proxy, priv),
            "proxy_raw_pose_corr": _corr(proxy, raw),
            "proxy_wrapped_pose_corr": _corr(proxy, wrapped),
            "raw_pose_wrapped_privileged_mae": float(np.nanmean([abs(_safe_float(r.get("pose_wrap_minus_privileged"))) for r in selected])) if selected else 0.0,
            "proxy_privileged_mae": float(np.nanmean([_safe_float(r.get("proxy_privileged_abs_error")) for r in selected])) if selected else 0.0,
            "proxy_neg_privileged_mae": float(np.nanmean([_safe_float(r.get("proxy_neg_privileged_abs_error")) for r in selected])) if selected else 0.0,
            "symmetry_aware_proxy_mae": float(symm_summary["mae"]),
            "symmetry_aware_proxy_bias": float(symm_bias),
            "symmetry_aware_proxy_bias_corrected_mae": float(symm_corrected_summary["mae"]),
            "symmetry_aware_proxy_bias_corrected_corr": float(symm_corrected_summary["corr"]),
            "symmetry_aware_proxy_bias_corrected_sign_match_rate": float(symm_corrected_summary["bias_corrected_sign_match_rate"]),
            "symmetry_aware_proxy_residual_std": float(symm_summary["residual_std"]),
            "symmetry_alias_better_rows": int(sum(
                np.isfinite(_safe_float(r.get("best_symmetry_alias_abs_error")))
                and _safe_float(r.get("best_symmetry_alias_abs_error")) + 1.0e-9 < _safe_float(r.get("proxy_privileged_abs_error"), float("inf"))
                and int(r.get("best_symmetry_alias_k", 0)) != 0
                for r in selected
            )),
        },
        "counts": {
            "by_episode": _group_counts(selected, "episode_idx"),
            "by_failure_bucket": _group_counts(selected, "failure_bucket"),
            "by_visual_observability": _group_counts(selected, "visual_observability_class"),
            "by_yaw_observability": _group_counts(selected, "yaw_observability_class"),
            "by_primary_blocker": _group_counts(selected, "yaw_observability_primary_blocker"),
            "by_blocker_combo": _group_counts(selected, "yaw_observability_blocker_combo"),
            "by_diagnosis_label": _group_counts(selected, "diagnosis_label"),
        },
        "by_episode": _group_metric(selected, "episode_idx"),
        "by_failure_bucket": _group_metric(selected, "failure_bucket"),
        "by_primary_blocker": _group_metric(selected, "yaw_observability_primary_blocker"),
        "by_diagnosis_label": _group_metric(selected, "diagnosis_label"),
        "top_abs_error_examples": sorted(
            selected,
            key=lambda r: _safe_float(r.get("proxy_privileged_abs_error"), -1.0),
            reverse=True,
        )[:50],
    }
    return selected, report


def main() -> None:
    ap = argparse.ArgumentParser(description="Diagnose yaw/frame alignment for C2C v2 frame residual relabels.")
    ap.add_argument("--relabel_jsonl", type=Path, required=True)
    ap.add_argument(
        "--output_dir",
        type=Path,
        default=Path("runtime_artifacts/coarse2contact_v2/reports/yaw_frame_alignment_diagnostic"),
    )
    ap.add_argument("--stage_name", type=str, default="RING_GRASP_ALIGN")
    ap.add_argument("--skill_type", type=str, default="precision_grasp")
    ap.add_argument("--visual_only", action="store_true", default=False)
    ap.add_argument("--near_basin_only", action="store_true", default=False)
    ap.add_argument("--min_frame_observability", type=float, default=None)
    ap.add_argument("--require_not_wrist_occluded", action="store_true", default=False)
    ap.add_argument("--default_symmetry_period", type=float, default=float(np.pi / 2.0))
    args = ap.parse_args()

    rows = _read_jsonl(args.relabel_jsonl)
    selected, report = diagnose(
        rows,
        stage_name=str(args.stage_name),
        skill_type=str(args.skill_type),
        visual_only=bool(args.visual_only),
        near_basin_only=bool(args.near_basin_only),
        min_frame_observability=args.min_frame_observability,
        require_not_wrist_occluded=bool(args.require_not_wrist_occluded),
        default_symmetry_period=float(args.default_symmetry_period),
    )

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = output_dir / "yaw_frame_diagnostic_rows.jsonl"
    with open(rows_path, "w", encoding="utf-8") as handle:
        for item in selected:
            handle.write(json.dumps(item, sort_keys=True) + "\n")

    report["source_jsonl"] = str(args.relabel_jsonl.resolve())
    report["diagnostic_rows_jsonl"] = str(rows_path)
    out_json = output_dir / "yaw_frame_alignment_diagnostic.json"
    out_md = output_dir / "yaw_frame_alignment_diagnostic.md"
    out_json.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    _write_plots(selected, output_dir)

    o = report["overall"]
    lines = [
        "# Yaw / Frame Alignment Diagnostic",
        "",
        f"- source: `{args.relabel_jsonl}`",
        f"- rows: `{o['num_rows']}`",
        f"- episodes: `{o['episodes']}`",
        f"- near_basin_shell_rows: `{o['near_basin_shell_rows']}`",
        f"- micro_entry_ready_rows: `{o['micro_entry_ready_rows']}`",
        f"- proxy_valid_rows: `{o['proxy_valid_rows']}`",
        f"- proxy_privileged_sign_match_rate: `{o['proxy_privileged_sign_match_rate']:.3f}`",
        f"- proxy_neg_privileged_sign_match_rate: `{o['proxy_neg_privileged_sign_match_rate']:.3f}`",
        f"- proxy_privileged_corr: `{o['proxy_privileged_corr']:.3f}`",
        f"- proxy_wrapped_pose_corr: `{o['proxy_wrapped_pose_corr']:.3f}`",
        f"- raw_pose_wrapped_privileged_mae: `{o['raw_pose_wrapped_privileged_mae']:.6f}`",
        f"- baseline: `-wrap(proxy, symmetry_period)` with symmetry_period=`{report['selection']['baseline_symmetry_period']:.6f}`",
        f"- symmetry_aware_proxy_mae: `{o['symmetry_aware_proxy_mae']:.6f}`",
        f"- symmetry_aware_proxy_bias: `{o['symmetry_aware_proxy_bias']:.6f}`",
        f"- symmetry_aware_proxy_bias_corrected_mae: `{o['symmetry_aware_proxy_bias_corrected_mae']:.6f}`",
        f"- symmetry_aware_proxy_bias_corrected_corr: `{o['symmetry_aware_proxy_bias_corrected_corr']:.3f}`",
        f"- symmetry_alias_better_rows: `{o['symmetry_alias_better_rows']}`",
        "",
        "## Diagnosis Counts",
    ]
    for name, count in report["counts"]["by_diagnosis_label"].items():
        lines.append(f"- `{name}`: `{count}`")
    lines.extend(["", "## Primary Blockers"])
    for name, count in report["counts"]["by_primary_blocker"].items():
        lines.append(f"- `{name}`: `{count}`")
    lines.extend(["", "## Failure Buckets"])
    for item in report["by_failure_bucket"]:
        lines.append(
            f"- `{item['failure_bucket']}`: rows={item['num_rows']}, sign={item['sign_match_rate']:.3f}, "
            f"corr={item['proxy_privileged_corr']:.3f}, mae={item['proxy_privileged_mae']:.3f}, diag={item['diagnosis_counts']}"
        )
        lines.extend(
        [
            "",
            "## Frame Rows",
            "",
            "| ep | step | xy | priv_yaw | symmetry_aware_proxy | bias_corrected | frame_obs | blocker | diagnosis |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
        ]
    )
    for item in sorted(selected, key=lambda r: (int(r.get("episode_idx", -1)), int(r.get("step_idx", -1))))[:80]:
        lines.append(
            "| "
            f"{int(item.get('episode_idx', -1)):03d} | "
            f"{int(item.get('step_idx', -1))} | "
            f"{_safe_float(item.get('xy_error')):.5f} | "
            f"{_safe_float(item.get('privileged_yaw')):.5f} | "
            f"{_safe_float(item.get('symmetry_aware_proxy_yaw')):.5f} | "
            f"{_safe_float(item.get('symmetry_aware_proxy_bias_corrected')):.5f} | "
            f"{_safe_float(item.get('frame_observability')):.5f} | "
            f"{item.get('yaw_observability_primary_blocker', '')} | "
            f"{item.get('diagnosis_label', '')} |"
        )
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps(report["overall"], indent=2, sort_keys=True))
    print(out_json)
    print(out_md)
    print(rows_path)


if __name__ == "__main__":
    main()
