#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from prismatic.models.student_candidate_evaluator_v2 import StudentCandidateEvaluatorV2
from scripts.build_pose_candidate_dataset import build_action_primitives, build_orientation_rescue_primitives, candidate_group_key


def find_trace_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if (path / "gripper_traces").is_dir():
        path = path / "gripper_traces"
    files = sorted(path.glob("*_gripper_trace.jsonl"))
    if not files:
        files = sorted(path.glob("*.jsonl"))
    return files


def load_rows(trace_dir: Path) -> list[dict]:
    rows: list[dict] = []
    for path in find_trace_files(trace_dir):
        with path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if bool(row.get("b2_candidate_shadow_gate_open", False)):
                    row["_trace_file"] = path.name
                    rows.append(row)
    return rows


def build_candidate_bank(yaw_probe_values: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    base_actions = build_action_primitives(
        xy_small=0.004,
        xy_large=0.008,
        z_small=0.004,
        yaw_small=0.03,
        yaw_probe_values=(),
        include_descend=True,
        include_combos=True,
        include_tilt=False,
    )
    rescue_actions = build_orientation_rescue_primitives(
        pitch_small=0.04,
        roll_small=0.04,
        xy_small=0.004,
        coupled_xy_tilt=True,
    )
    actions = list(base_actions) + list(rescue_actions)
    mask = [1.0] * len(base_actions) + [0.0] * len(rescue_actions)
    for text in str(yaw_probe_values or "").split(","):
        text = text.strip()
        if not text:
            continue
        mag = abs(float(text))
        if mag <= 0.0:
            continue
        pos = np.zeros((6,), dtype=np.float32)
        neg = np.zeros((6,), dtype=np.float32)
        pos[5] = mag
        neg[5] = -mag
        actions.extend([pos, neg])
        mask.extend([1.0, 1.0])
    actions_np = np.stack(actions, axis=0).astype(np.float32)
    group_keys = [candidate_group_key(c) for c in actions_np]
    unique = {key: idx for idx, key in enumerate(sorted(set(group_keys)))}
    groups = np.asarray([unique[key] for key in group_keys], dtype=np.int64)
    return actions_np, np.asarray(mask, dtype=np.float32), groups


def sign_bucket(v: float, eps: float) -> int:
    if v > eps:
        return 2
    if v < -eps:
        return 0
    return 1


def make_dataset(rows: list[dict], actions: np.ndarray, mask: np.ndarray, label_policy: str) -> dict[str, np.ndarray]:
    deltas = []
    labels = []
    valid = []
    confidence = []
    episode = []
    step = []
    meta_rows = []
    apply_label = 2
    for row in rows:
        delta = np.asarray(row.get("refiner_current_delta_basin_target", row.get("current_delta_basin_target", [0, 0, 0, 0, 0, 0])), dtype=np.float32).reshape(-1)
        if delta.size < 6:
            continue
        yaw_needed = bool(row.get("b2_candidate_shadow_yaw_needed", False))
        yaw_keep = bool(row.get("b2_candidate_shadow_yaw_keep", False))
        xy_block = bool(row.get("b2_candidate_shadow_xy_block", False))
        teacher_ready = bool(row.get("b2_candidate_shadow_teacher_ready", False))
        if label_policy == "yaw_needed_not_blocked":
            is_apply = bool(yaw_needed and not yaw_keep and not xy_block and not teacher_ready)
        elif label_policy == "runtime_yaw_threshold":
            handoff = row.get("handoff_release_metric_thresholds_provider", {}) or {}
            yaw_thr = float(handoff.get("yaw_error", 0.12434040009975433))
            is_apply = bool(abs(float(delta[5])) > yaw_thr and not xy_block and not teacher_ready)
        else:
            raise ValueError(f"unknown label policy: {label_policy}")
        labels.append(apply_label if is_apply else 0)
        valid.append(1.0)
        # Emphasize hard anchors, but keep low-confidence rows usable.
        margin = abs(float(delta[5])) / 0.12434040009975433
        confidence.append(float(np.clip(0.5 + 0.5 * margin, 0.5, 2.0)))
        deltas.append(delta[:6].astype(np.float32))
        episode.append(int(str(row.get("_trace_file", "ep000")).split("ep", 1)[-1].split("_", 1)[0]))
        step.append(int(row.get("step", -1)))
        meta_rows.append(row)
    if not deltas:
        raise RuntimeError("no usable B2 shadow rows found")
    n = len(deltas)
    out = {
        "proxy_current_delta_basin_target": np.stack(deltas, axis=0).astype(np.float32),
        "candidate_actions_local": np.repeat(actions[None, :, :], n, axis=0).astype(np.float32),
        "candidate_mask": np.repeat(mask[None, :], n, axis=0).astype(np.float32),
        "b2_yaw_aware_candidate_scope_v3": np.repeat(mask[None, :], n, axis=0).astype(np.float32),
        "yaw_mode3_label_v11": np.asarray(labels, dtype=np.int64),
        "yaw_mode_valid_v11": np.asarray(valid, dtype=np.float32),
        "yaw_mode_confidence_v11": np.asarray(confidence, dtype=np.float32),
        "episode_index": np.asarray(episode, dtype=np.int64),
        "step_idx": np.asarray(step, dtype=np.int64),
        "current_dx_sign": np.asarray([sign_bucket(float(d[0]), 1e-4) for d in deltas], dtype=np.int64),
        "current_dy_sign": np.asarray([sign_bucket(float(d[1]), 1e-4) for d in deltas], dtype=np.int64),
        "current_dyaw_sign": np.asarray([sign_bucket(float(d[5]), 1e-3) for d in deltas], dtype=np.int64),
        "basin_distance_bin": np.asarray([3] * n, dtype=np.int64),
        "runtime_shadow_pred_regret": np.asarray([float(r.get("b2_candidate_shadow_pred_regret", np.nan)) for r in meta_rows], dtype=np.float32),
        "runtime_shadow_baseline_regret": np.asarray([float(r.get("b2_candidate_shadow_baseline_regret", np.nan)) for r in meta_rows], dtype=np.float32),
        "runtime_shadow_regret_delta": np.asarray([float(r.get("b2_candidate_shadow_regret_delta", np.nan)) for r in meta_rows], dtype=np.float32),
        "runtime_shadow_yaw_needed": np.asarray([float(bool(r.get("b2_candidate_shadow_yaw_needed", False))) for r in meta_rows], dtype=np.float32),
        "runtime_shadow_yaw_keep": np.asarray([float(bool(r.get("b2_candidate_shadow_yaw_keep", False))) for r in meta_rows], dtype=np.float32),
        "runtime_shadow_xy_block": np.asarray([float(bool(r.get("b2_candidate_shadow_xy_block", False))) for r in meta_rows], dtype=np.float32),
    }
    return out


def balanced_accuracy(pred: np.ndarray, target: np.ndarray) -> dict:
    keep = target == 0
    apply = target == 2
    keep_recall = float(np.mean(pred[keep] == 0)) if np.any(keep) else math.nan
    apply_recall = float(np.mean(pred[apply] == 2)) if np.any(apply) else math.nan
    vals = [v for v in (keep_recall, apply_recall) if math.isfinite(v)]
    return {
        "keep_recall": keep_recall,
        "apply_recall": apply_recall,
        "balanced_acc": float(np.mean(vals)) if vals else math.nan,
        "keep_count": int(np.sum(keep)),
        "apply_count": int(np.sum(apply)),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace_dir", type=Path, required=True)
    ap.add_argument("--base_ckpt", type=Path, required=True)
    ap.add_argument("--output_dir", type=Path, required=True)
    ap.add_argument("--yaw_probe_values", type=str, default="0.06,0.12")
    ap.add_argument("--label_policy", type=str, default="yaw_needed_not_blocked", choices=["yaw_needed_not_blocked", "runtime_yaw_threshold"])
    ap.add_argument("--val_episodes", type=str, default="18,34,45")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=3407)
    args = ap.parse_args()

    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_rows(args.trace_dir)
    actions, mask, groups = build_candidate_bank(args.yaw_probe_values)
    data = make_dataset(rows, actions, mask, args.label_policy)
    np.savez_compressed(args.output_dir / "b2_runtime_shadow_v14c_summary_mode_dataset.npz", **data, candidate_group_index=np.repeat(groups[None, :], data["episode_index"].shape[0], axis=0))

    ckpt = torch.load(args.base_ckpt, map_location="cpu")
    model = StudentCandidateEvaluatorV2(yaw_mode_classes=int(ckpt.get("yaw_mode_num_classes", 3)))
    model.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=False)
    model.set_mode_input_path("summary_only")
    for p in model.parameters():
        p.requires_grad = False
    for name in ("delta_encoder", "candidate_summary_encoder", "summary_context_head", "yaw_mode_head"):
        for p in getattr(model, name).parameters():
            p.requires_grad = True
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device).train()

    val_eps = {int(x) for x in str(args.val_episodes).split(",") if x.strip()}
    ep = data["episode_index"]
    train_idx = np.asarray([i for i, e in enumerate(ep.tolist()) if int(e) not in val_eps], dtype=np.int64)
    val_idx = np.asarray([i for i, e in enumerate(ep.tolist()) if int(e) in val_eps], dtype=np.int64)
    if train_idx.size == 0 or val_idx.size == 0:
        rng = np.random.default_rng(int(args.seed))
        idx = np.arange(ep.shape[0])
        rng.shuffle(idx)
        cut = max(1, int(round(idx.size * 0.75)))
        train_idx, val_idx = idx[:cut], idx[cut:]

    x_delta = torch.from_numpy(data["proxy_current_delta_basin_target"]).to(device)
    x_actions = torch.from_numpy(data["candidate_actions_local"]).to(device)
    x_mask = torch.from_numpy(data["candidate_mask"]).to(device)
    y = torch.from_numpy(data["yaw_mode3_label_v11"]).long().to(device)
    conf = torch.from_numpy(data["yaw_mode_confidence_v11"]).float().to(device)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=float(args.lr), weight_decay=1e-4)

    history = []
    best = None
    best_state = None
    batch_size = 256
    rng = np.random.default_rng(int(args.seed))
    latent = torch.zeros((x_delta.shape[0], 128), device=device)
    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        order = train_idx.copy()
        rng.shuffle(order)
        losses = []
        for start in range(0, order.size, batch_size):
            idx = torch.from_numpy(order[start : start + batch_size]).long().to(device)
            out = model.forward_with_mode(
                handoff_latent=latent[idx],
                proxy_current_delta_basin_target=x_delta[idx],
                candidate_actions_local=x_actions[idx],
                candidate_mask=x_mask[idx],
                yaw_aware_candidate_scope=x_mask[idx],
            )
            logits = out["yaw_mode_logits"]
            target = y[idx]
            weights = torch.ones_like(target, dtype=torch.float32)
            for cls in (0, 2):
                cls_mask = target == cls
                if torch.any(cls_mask):
                    weights[cls_mask] = float(target.numel()) / (2.0 * float(cls_mask.sum().item()))
            weights = weights * torch.clamp(conf[idx], 0.5, 2.0)
            loss_vec = F.cross_entropy(logits, target, reduction="none")
            loss = (loss_vec * weights).sum() / torch.clamp(weights.sum(), min=1e-6)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))
        model.eval()
        with torch.no_grad():
            def eval_idx(arr):
                idx = torch.from_numpy(arr).long().to(device)
                out = model.forward_with_mode(
                    handoff_latent=latent[idx],
                    proxy_current_delta_basin_target=x_delta[idx],
                    candidate_actions_local=x_actions[idx],
                    candidate_mask=x_mask[idx],
                    yaw_aware_candidate_scope=x_mask[idx],
                )
                pred = torch.argmax(out["yaw_mode_logits"], dim=-1).detach().cpu().numpy()
                return balanced_accuracy(pred, y[idx].detach().cpu().numpy())
            tr = eval_idx(train_idx)
            va = eval_idx(val_idx)
        row = {"epoch": epoch, "loss": float(np.mean(losses)) if losses else math.nan, "train": tr, "val": va}
        history.append(row)
        metric = va["balanced_acc"]
        if best is None or (math.isfinite(metric) and metric > best["val"]["balanced_acc"]):
            best = row
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    out_ckpt = dict(ckpt)
    out_ckpt["model_state_dict"] = best_state
    out_ckpt["mode_input_path"] = "summary_only"
    out_ckpt["mode_feature_version"] = "runtime_shadow_v14c_summary_mode"
    out_ckpt["runtime_shadow_v14c"] = {
        "trace_dir": str(args.trace_dir),
        "label_policy": args.label_policy,
        "yaw_probe_values": args.yaw_probe_values,
        "train_rows": int(train_idx.size),
        "val_rows": int(val_idx.size),
        "best_epoch": int(best["epoch"]),
        "best_val": best["val"],
    }
    torch.save(out_ckpt, args.output_dir / "student_candidate_evaluator_v2_v14c_runtime_summary_mode.pt")
    report = {
        "dataset_npz": str(args.output_dir / "b2_runtime_shadow_v14c_summary_mode_dataset.npz"),
        "checkpoint": str(args.output_dir / "student_candidate_evaluator_v2_v14c_runtime_summary_mode.pt"),
        "label_policy": args.label_policy,
        "rows": int(ep.shape[0]),
        "label_counts": {str(k): int(v) for k, v in zip(*np.unique(data["yaw_mode3_label_v11"], return_counts=True))},
        "train_rows": int(train_idx.size),
        "val_rows": int(val_idx.size),
        "val_episodes": sorted(int(x) for x in set(ep[val_idx].tolist())),
        "best": best,
        "history": history,
    }
    (args.output_dir / "runtime_summary_mode_train_report.json").write_text(json.dumps(report, indent=2, allow_nan=True))
    print(json.dumps({k: report[k] for k in ("rows", "label_counts", "train_rows", "val_rows", "best", "checkpoint")}, indent=2, allow_nan=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
