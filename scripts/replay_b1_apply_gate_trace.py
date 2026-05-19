#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from build_b1_apply_gate_dataset import find_trace_files, row_features
from train_b1_apply_gate import ApplyGateMLP


def as_float(row: dict, key: str, default: float = math.nan) -> float:
    try:
        value = float(row.get(key, default))
    except Exception:
        return default
    return value if math.isfinite(value) else default


def load_model(path: Path):
    ckpt = torch.load(path, map_location="cpu")
    hidden_dim = int(ckpt.get("hidden_dim", 32))
    feature_mean = np.asarray(ckpt["feature_mean"], dtype=np.float32)
    feature_std = np.asarray(ckpt["feature_std"], dtype=np.float32)
    feature_names = [str(v) for v in ckpt.get("feature_names", [])]
    model = ApplyGateMLP(feature_mean.shape[1], hidden_dim=hidden_dim)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    threshold = float(ckpt.get("threshold", 0.5))
    return {
        "model": model,
        "feature_mean": feature_mean,
        "feature_std": feature_std,
        "threshold": threshold,
        "feature_names": feature_names,
        "selection_rule": str(ckpt.get("selection_rule", "threshold_only")),
        "veto_runtime_yaw_norm_gt": float(ckpt.get("veto_runtime_yaw_norm_gt", np.inf)),
        "veto_pred_uncertainty_gt": float(ckpt.get("veto_pred_uncertainty_gt", np.inf)),
        "veto_group_margin_lt": float(ckpt.get("veto_group_margin_lt", -np.inf)),
        "veto_baseline_groups": [int(v) for v in ckpt.get("veto_baseline_groups", [])],
        "require_runtime_yaw_norm_ge": float(ckpt.get("require_runtime_yaw_norm_ge", -np.inf)),
        "require_runtime_yaw_norm_le": float(ckpt.get("require_runtime_yaw_norm_le", np.inf)),
        "require_group_margin_ge": float(ckpt.get("require_group_margin_ge", -np.inf)),
        "require_uncertainty_ge_or_yaw_norm_le": [
            float(v) for v in ckpt.get("require_uncertainty_ge_or_yaw_norm_le", [])
        ],
        "veto_pred_group_if_yaw_norm_le_and_uncertainty_lt": [
            float(v) for v in ckpt.get("veto_pred_group_if_yaw_norm_le_and_uncertainty_lt", [])
        ],
    }


def summarize(selected: list[float], all_deltas: list[float]) -> dict:
    selected_arr = np.asarray(selected, dtype=np.float32)
    all_arr = np.asarray(all_deltas, dtype=np.float32)
    negative = selected_arr[selected_arr <= 0.0]
    positive = selected_arr[selected_arr > 0.0]
    return {
        "valid_frames": int(all_arr.size),
        "apply_count": int(selected_arr.size),
        "apply_rate": float(selected_arr.size / max(all_arr.size, 1)),
        "positive_apply_count": int(positive.size),
        "negative_apply_count": int(negative.size),
        "negative_apply_rate": float(negative.size / max(selected_arr.size, 1)),
        "positive_apply_rate": float(positive.size / max(selected_arr.size, 1)),
        "mean_regret_delta_selected": float(np.mean(selected_arr)) if selected_arr.size else 0.0,
        "mean_regret_delta_all": float(np.mean(all_arr)) if all_arr.size else 0.0,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace_dir", required=True)
    ap.add_argument("--gate_ckpt", required=True)
    ap.add_argument("--output_json", required=True)
    ap.add_argument("--gate_mode", choices=["close_only", "all_gate"], default="close_only")
    args = ap.parse_args()

    cfg = load_model(Path(args.gate_ckpt))
    model = cfg["model"]
    mean = cfg["feature_mean"]
    std = cfg["feature_std"]
    threshold = cfg["threshold"]
    feature_names = cfg["feature_names"]
    files = find_trace_files(Path(args.trace_dir))
    selected, all_deltas = [], []
    by_episode = {}
    for file in files:
        ep_selected, ep_all = [], []
        with file.open() as f:
            for line in f:
                row = json.loads(line)
                if not row.get("b1_group_shadow_gate_open", False):
                    continue
                if args.gate_mode == "close_only" and not row.get("b1_group_shadow_close_neighborhood", False):
                    continue
                regret_delta = as_float(row, "b1_group_shadow_regret_delta")
                if not math.isfinite(regret_delta):
                    continue
                feat = np.asarray(row_features(row, feature_names if feature_names else None), dtype=np.float32)[None, :]
                feat = (feat - mean) / std
                with torch.no_grad():
                    prob = float(torch.sigmoid(model(torch.from_numpy(feat))).item())
                all_deltas.append(regret_delta)
                ep_all.append(regret_delta)
                apply = prob >= threshold
                if apply and cfg["selection_rule"] == "yaw_uncertainty_veto":
                    runtime_yaw_norm = float(row.get("runtime_handoff_yaw_norm", math.nan))
                    if not math.isfinite(runtime_yaw_norm):
                        aux = row.get("handoff_aux_provider") or {}
                        hm = row.get("handoff_metrics_provider") or {}
                        rel = row.get("handoff_release_metric_thresholds_provider") or {}
                        try:
                            runtime_yaw_norm = float(hm.get("yaw_error", math.nan)) / max(
                                float(rel.get("yaw_error", 0.12434040009975433)),
                                1e-6,
                            )
                        except Exception:
                            runtime_yaw_norm = math.nan
                    pred_unc = math.nan
                    aux = row.get("handoff_aux_provider") or {}
                    try:
                        pred_unc = float(aux.get("pred_uncertainty", math.nan))
                    except Exception:
                        pred_unc = math.nan
                    if not (math.isfinite(runtime_yaw_norm) and math.isfinite(pred_unc)):
                        apply = False
                    elif (
                        runtime_yaw_norm > cfg["veto_runtime_yaw_norm_gt"]
                        and pred_unc > cfg["veto_pred_uncertainty_gt"]
                    ):
                        apply = False
                if apply and cfg["selection_rule"] == "close_yawaware_v6":
                    aux = row.get("handoff_aux_provider") or {}
                    hm = row.get("handoff_metrics_provider") or {}
                    rel = row.get("handoff_release_metric_thresholds_provider") or {}
                    runtime_yaw_norm = float(row.get("runtime_handoff_yaw_norm", math.nan))
                    if not math.isfinite(runtime_yaw_norm):
                        try:
                            runtime_yaw_norm = float(hm.get("yaw_error", math.nan)) / max(
                                float(rel.get("yaw_error", 0.12434040009975433)),
                                1e-6,
                            )
                        except Exception:
                            runtime_yaw_norm = math.nan
                    try:
                        pred_unc = float(aux.get("pred_uncertainty", math.nan))
                    except Exception:
                        pred_unc = math.nan
                    margin = float(row.get("b1_group_shadow_margin", math.nan))
                    if not (math.isfinite(runtime_yaw_norm) and math.isfinite(pred_unc) and math.isfinite(margin)):
                        apply = False
                    elif runtime_yaw_norm < cfg["require_runtime_yaw_norm_ge"]:
                        apply = False
                    elif runtime_yaw_norm > cfg["require_runtime_yaw_norm_le"]:
                        apply = False
                    elif margin < cfg["require_group_margin_ge"]:
                        apply = False
                    else:
                        pair = cfg.get("require_uncertainty_ge_or_yaw_norm_le", [])
                        if pair and len(pair) >= 2 and not (pred_unc >= pair[0] or runtime_yaw_norm <= pair[1]):
                            apply = False
                    veto = cfg.get("veto_pred_group_if_yaw_norm_le_and_uncertainty_lt", [])
                    if apply and veto and len(veto) >= 3:
                        try:
                            pred_group = int(row.get("b1_group_shadow_pred_group", -1))
                        except Exception:
                            pred_group = -1
                        if pred_group == int(veto[0]) and runtime_yaw_norm <= veto[1] and pred_unc < veto[2]:
                            apply = False
                veto_group_margin_lt = cfg.get("veto_group_margin_lt", -math.inf)
                margin = float(row.get("b1_group_shadow_margin", math.nan))
                if apply and math.isfinite(veto_group_margin_lt):
                    if not math.isfinite(margin) or margin < veto_group_margin_lt:
                        apply = False
                veto_baseline_groups = cfg.get("veto_baseline_groups", [])
                if apply and veto_baseline_groups:
                    try:
                        baseline_group = int(row.get("b1_group_shadow_baseline_group", -1))
                    except Exception:
                        baseline_group = -1
                    if baseline_group in set(int(v) for v in veto_baseline_groups):
                        apply = False
                if apply:
                    selected.append(regret_delta)
                    ep_selected.append(regret_delta)
        by_episode[file.name] = summarize(ep_selected, ep_all)

    report = {
        "trace_dir": args.trace_dir,
        "gate_ckpt": args.gate_ckpt,
        "threshold": threshold,
        "summary": summarize(selected, all_deltas),
        "episodes": by_episode,
    }
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
