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
from scripts.build_pose_candidate_dataset import (
    build_action_primitives,
    build_orientation_rescue_primitives,
    candidate_group_key,
)


def find_trace_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if (path / "gripper_traces").is_dir():
        path = path / "gripper_traces"
    files = sorted(path.glob("*_gripper_trace.jsonl"))
    if not files:
        files = sorted(path.glob("*.jsonl"))
    return files


def episode_from_path(path: Path) -> int:
    name = path.name
    if "ep" in name:
        tail = name.split("ep", 1)[-1]
        digits = "".join(ch for ch in tail[:4] if ch.isdigit())
        if digits:
            return int(digits)
    return -1


def load_rows(trace_dir: Path) -> list[dict]:
    rows: list[dict] = []
    for path in find_trace_files(trace_dir):
        ep = episode_from_path(path)
        with path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if bool(row.get("b2_candidate_shadow_gate_open", False)):
                    row["_episode_index"] = int(ep)
                    row["_trace_file"] = path.name
                    rows.append(row)
    if not rows:
        raise RuntimeError(f"no B2 gate-open rows found under {trace_dir}")
    return rows


def fallback_candidate_bank(yaw_probe_values: str) -> tuple[np.ndarray, np.ndarray]:
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
    scope = [1.0] * len(base_actions) + [0.0] * len(rescue_actions)
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
        scope.extend([1.0, 1.0])
    return np.stack(actions, axis=0).astype(np.float32), np.asarray(scope, dtype=np.float32)


def sign_bucket(v: float, eps: float) -> int:
    if v > eps:
        return 2
    if v < -eps:
        return 0
    return 1


def choose_label(
    row: dict,
    actions: np.ndarray,
    scope: np.ndarray,
    cost: np.ndarray | None,
    *,
    keep_yaw_abs: float,
    apply_cost_margin: float,
    label_policy: str,
) -> int:
    yaw_needed = bool(row.get("b2_candidate_shadow_yaw_needed", False))
    yaw_keep = bool(row.get("b2_candidate_shadow_yaw_keep", False))
    xy_block = bool(row.get("b2_candidate_shadow_xy_block", False))
    teacher_ready = bool(row.get("b2_candidate_shadow_teacher_ready", False))
    teacher_bucket_label = 2 if bool(yaw_needed and not yaw_keep and not xy_block and not teacher_ready) else 0
    if label_policy == "teacher_bucket":
        return teacher_bucket_label
    if cost is not None and np.any(np.isfinite(cost)):
        yaw_abs = np.abs(actions[:, 5])
        valid = (scope > 0.5) & np.isfinite(cost)
        keep = valid & (yaw_abs <= keep_yaw_abs)
        apply = valid & (yaw_abs > keep_yaw_abs)
        keep_best = float(np.min(cost[keep])) if np.any(keep) else math.inf
        apply_best = float(np.min(cost[apply])) if np.any(apply) else math.inf
        if (
            math.isfinite(apply_best)
            and apply_best + float(apply_cost_margin) < keep_best
            and not xy_block
            and not teacher_ready
        ):
            return 2
        if label_policy == "cost_anchor":
            return 0
    return teacher_bucket_label


def make_dataset(
    rows: list[dict],
    *,
    yaw_probe_values: str,
    keep_yaw_abs: float,
    apply_cost_margin: float,
    label_policy: str,
) -> dict[str, np.ndarray | list[str]]:
    fallback_actions, fallback_scope = fallback_candidate_bank(yaw_probe_values)
    deltas, actions_all, scope_all, valid_all, costs_all = [], [], [], [], []
    labels, confidence, episodes, steps, trace_files = [], [], [], [], []
    full_rank_rows = 0
    for row in rows:
        delta = np.asarray(
            row.get("refiner_current_delta_basin_target", row.get("current_delta_basin_target", [0, 0, 0, 0, 0, 0])),
            dtype=np.float32,
        ).reshape(-1)
        if delta.size < 6:
            continue
        actions = row.get("b2_candidate_shadow_candidate_actions_local", None)
        if actions is None:
            actions_np = fallback_actions.copy()
            scope_np = fallback_scope.copy()
            valid_np = np.ones((actions_np.shape[0],), dtype=np.float32)
            cost_np = np.full((actions_np.shape[0],), np.nan, dtype=np.float32)
        else:
            actions_np = np.asarray(actions, dtype=np.float32)
            if actions_np.ndim != 2 or actions_np.shape[1] < 6:
                continue
            actions_np = actions_np[:, :6].astype(np.float32)
            scope_np = np.asarray(
                row.get("b2_candidate_shadow_candidate_scope_mask", np.ones((actions_np.shape[0],), dtype=np.float32)),
                dtype=np.float32,
            ).reshape(-1)
            valid_np = np.asarray(
                row.get("b2_candidate_shadow_candidate_valid_mask", scope_np),
                dtype=np.float32,
            ).reshape(-1)
            raw_cost = row.get("b2_candidate_shadow_candidate_cost", None)
            cost_np = (
                np.asarray(raw_cost, dtype=np.float32).reshape(-1)
                if raw_cost is not None
                else np.full((actions_np.shape[0],), np.nan, dtype=np.float32)
            )
            if cost_np.shape[0] == actions_np.shape[0] and np.any(np.isfinite(cost_np)):
                full_rank_rows += 1
        n = actions_np.shape[0]
        if scope_np.shape[0] != n or valid_np.shape[0] != n or cost_np.shape[0] != n:
            continue
        label = choose_label(
            row,
            actions_np,
            scope_np,
            cost_np if np.any(np.isfinite(cost_np)) else None,
            keep_yaw_abs=keep_yaw_abs,
            apply_cost_margin=apply_cost_margin,
            label_policy=label_policy,
        )
        deltas.append(delta[:6].astype(np.float32))
        actions_all.append(actions_np)
        scope_all.append(scope_np.astype(np.float32))
        valid_all.append(valid_np.astype(np.float32))
        costs_all.append(cost_np.astype(np.float32))
        labels.append(int(label))
        confidence.append(float(np.clip(0.5 + 0.5 * abs(float(delta[5])) / 0.12434040009975433, 0.5, 2.0)))
        episodes.append(int(row.get("_episode_index", -1)))
        steps.append(int(row.get("step", -1)))
        trace_files.append(str(row.get("_trace_file", "")))
    if not deltas:
        raise RuntimeError("no usable runtime B2 rows after filtering")
    lengths = {a.shape[0] for a in actions_all}
    if len(lengths) != 1:
        raise RuntimeError(f"variable candidate bank lengths are not supported yet: {sorted(lengths)}")
    cost = np.stack(costs_all, axis=0).astype(np.float32)
    oracle = -cost
    oracle[~np.isfinite(oracle)] = -1.0e9
    best = np.argmax(oracle, axis=1).astype(np.int64)
    return {
        "proxy_current_delta_basin_target": np.stack(deltas, axis=0).astype(np.float32),
        "candidate_actions_local": np.stack(actions_all, axis=0).astype(np.float32),
        "candidate_mask": np.stack(valid_all, axis=0).astype(np.float32),
        "b2_yaw_aware_candidate_scope_v3": np.stack(scope_all, axis=0).astype(np.float32),
        "candidate_oracle_score": oracle.astype(np.float32),
        "best_candidate_index": best,
        "b2_yaw_aware_best_candidate_index_v3": best,
        "yaw_mode3_label_v11": np.asarray(labels, dtype=np.int64),
        "yaw_mode_valid_v11": np.ones((len(labels),), dtype=np.float32),
        "yaw_mode_confidence_v11": np.asarray(confidence, dtype=np.float32),
        "episode_index": np.asarray(episodes, dtype=np.int64),
        "step_idx": np.asarray(steps, dtype=np.int64),
        "current_dx_sign": np.asarray([sign_bucket(float(d[0]), 1e-4) for d in deltas], dtype=np.int64),
        "current_dy_sign": np.asarray([sign_bucket(float(d[1]), 1e-4) for d in deltas], dtype=np.int64),
        "current_dyaw_sign": np.asarray([sign_bucket(float(d[5]), 1e-3) for d in deltas], dtype=np.int64),
        "basin_distance_bin": np.asarray([3] * len(labels), dtype=np.int64),
        "trace_file": np.asarray(trace_files),
        "full_rank_row_mask": np.asarray(np.any(np.isfinite(cost), axis=1), dtype=np.float32),
        "_full_rank_rows": int(full_rank_rows),
    }


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
    ap.add_argument("--val_episodes", type=str, default="18,34,45")
    ap.add_argument("--mode_epochs", type=int, default=40)
    ap.add_argument("--rank_epochs", type=int, default=0)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--rank_lr", type=float, default=1e-4)
    ap.add_argument("--keep_yaw_abs", type=float, default=0.035)
    ap.add_argument("--apply_cost_margin", type=float, default=0.02)
    ap.add_argument(
        "--mode_label_policy",
        type=str,
        default="teacher_bucket",
        choices=["teacher_bucket", "cost_anchor", "hybrid"],
    )
    ap.add_argument("--seed", type=int, default=3407)
    args = ap.parse_args()

    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_rows(args.trace_dir)
    data = make_dataset(
        rows,
        yaw_probe_values=args.yaw_probe_values,
        keep_yaw_abs=float(args.keep_yaw_abs),
        apply_cost_margin=float(args.apply_cost_margin),
        label_policy=str(args.mode_label_policy),
    )
    full_rank_rows = int(data.pop("_full_rank_rows"))
    np.savez_compressed(args.output_dir / "b2_runtime_trace_v14e_dataset.npz", **data)

    ckpt = torch.load(args.base_ckpt, map_location="cpu")
    model = StudentCandidateEvaluatorV2(yaw_mode_classes=int(ckpt.get("yaw_mode_num_classes", 3)))
    model.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=False)
    model.set_mode_input_path("summary_only")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    ep = np.asarray(data["episode_index"], dtype=np.int64)
    val_eps = {int(x) for x in str(args.val_episodes).split(",") if x.strip()}
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
    x_scope = torch.from_numpy(data["b2_yaw_aware_candidate_scope_v3"]).to(device)
    y = torch.from_numpy(data["yaw_mode3_label_v11"]).long().to(device)
    conf = torch.from_numpy(data["yaw_mode_confidence_v11"]).float().to(device)
    latent = torch.zeros((x_delta.shape[0], 128), device=device)
    rng = np.random.default_rng(int(args.seed))

    for p in model.parameters():
        p.requires_grad = False
    for name in ("delta_encoder", "candidate_summary_encoder", "summary_context_head", "yaw_mode_head"):
        for p in getattr(model, name).parameters():
            p.requires_grad = True
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=float(args.lr), weight_decay=1e-4)

    history = []
    best = None
    best_state = None
    batch_size = 256
    for epoch in range(1, int(args.mode_epochs) + 1):
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
                yaw_aware_candidate_scope=x_scope[idx],
            )
            target = y[idx]
            weights = torch.ones_like(target, dtype=torch.float32)
            for cls in (0, 2):
                cls_mask = target == cls
                if torch.any(cls_mask):
                    weights[cls_mask] = float(target.numel()) / (2.0 * float(cls_mask.sum().item()))
            weights = weights * torch.clamp(conf[idx], 0.5, 2.0)
            loss_vec = F.cross_entropy(out["yaw_mode_logits"], target, reduction="none")
            loss = (loss_vec * weights).sum() / torch.clamp(weights.sum(), min=1e-6)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))
        model.eval()
        with torch.no_grad():
            eval_rows = {}
            for name, arr in (("train", train_idx), ("val", val_idx)):
                idx = torch.from_numpy(arr).long().to(device)
                out = model.forward_with_mode(
                    handoff_latent=latent[idx],
                    proxy_current_delta_basin_target=x_delta[idx],
                    candidate_actions_local=x_actions[idx],
                    candidate_mask=x_mask[idx],
                    yaw_aware_candidate_scope=x_scope[idx],
                )
                pred = torch.argmax(out["yaw_mode_logits"], dim=-1).detach().cpu().numpy()
                eval_rows[name] = balanced_accuracy(pred, y[idx].detach().cpu().numpy())
        row = {"stage": "mode", "epoch": epoch, "loss": float(np.mean(losses)) if losses else math.nan, **eval_rows}
        history.append(row)
        metric = float(row["val"]["balanced_acc"])
        if best is None or (math.isfinite(metric) and metric > float(best["val"]["balanced_acc"])):
            best = row
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state, strict=False)

    rank_history = []
    if int(args.rank_epochs) > 0 and full_rank_rows > 0:
        rank_mask = np.asarray(data["full_rank_row_mask"], dtype=np.float32) > 0.5
        rank_train_idx = np.asarray([i for i in train_idx.tolist() if rank_mask[i]], dtype=np.int64)
        rank_val_idx = np.asarray([i for i in val_idx.tolist() if rank_mask[i]], dtype=np.int64)
        oracle = torch.from_numpy(data["candidate_oracle_score"]).to(device)
        best_idx = torch.from_numpy(data["best_candidate_index"]).long().to(device)
        for p in model.parameters():
            p.requires_grad = False
        for name in ("delta_encoder", "action_encoder", "keep_yaw_score_head", "apply_yaw_score_head"):
            for p in getattr(model, name).parameters():
                p.requires_grad = True
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=float(args.rank_lr), weight_decay=1e-4)
        for epoch in range(1, int(args.rank_epochs) + 1):
            model.train()
            order = rank_train_idx.copy()
            rng.shuffle(order)
            losses = []
            for start in range(0, order.size, 128):
                idx = torch.from_numpy(order[start : start + 128]).long().to(device)
                out = model.forward_with_mode(
                    handoff_latent=latent[idx],
                    proxy_current_delta_basin_target=x_delta[idx],
                    candidate_actions_local=x_actions[idx],
                    candidate_mask=x_mask[idx],
                    yaw_aware_candidate_scope=x_scope[idx],
                )
                apply_rows = y[idx] == 2
                scores = torch.where(apply_rows[:, None], out["candidate_scores_apply"], out["candidate_scores_keep"])
                scores = scores.masked_fill(x_scope[idx] <= 0.5, -1e9)
                target = best_idx[idx]
                loss = F.cross_entropy(scores, target)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                losses.append(float(loss.item()))
            rank_history.append({"stage": "rank", "epoch": epoch, "loss": float(np.mean(losses)) if losses else math.nan})

    out_ckpt = dict(ckpt)
    out_ckpt["model_state_dict"] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    out_ckpt["mode_input_path"] = "summary_only"
    out_ckpt["mode_feature_version"] = "runtime_trace_v14e_summary_first"
    out_ckpt["runtime_trace_v14e"] = {
        "trace_dir": str(args.trace_dir),
        "yaw_probe_values": args.yaw_probe_values,
        "keep_yaw_abs": float(args.keep_yaw_abs),
        "apply_cost_margin": float(args.apply_cost_margin),
        "mode_label_policy": str(args.mode_label_policy),
        "rows": int(ep.shape[0]),
        "full_rank_rows": int(full_rank_rows),
        "train_rows": int(train_idx.size),
        "val_rows": int(val_idx.size),
        "best_mode": best,
        "rank_epochs": int(args.rank_epochs),
    }
    ckpt_path = args.output_dir / "student_candidate_evaluator_v2_v14e_runtime_trace.pt"
    torch.save(out_ckpt, ckpt_path)

    label_keys, label_counts = np.unique(np.asarray(data["yaw_mode3_label_v11"], dtype=np.int64), return_counts=True)
    report = {
        "dataset_npz": str(args.output_dir / "b2_runtime_trace_v14e_dataset.npz"),
        "checkpoint": str(ckpt_path),
        "trace_dir": str(args.trace_dir),
        "mode_label_policy": str(args.mode_label_policy),
        "rows": int(ep.shape[0]),
        "full_rank_rows": int(full_rank_rows),
        "label_counts": {str(int(k)): int(v) for k, v in zip(label_keys, label_counts)},
        "train_rows": int(train_idx.size),
        "val_rows": int(val_idx.size),
        "val_episodes": sorted(int(x) for x in set(ep[val_idx].tolist())),
        "best_mode": best,
        "history": history,
        "rank_history": rank_history,
    }
    (args.output_dir / "runtime_trace_v14e_train_report.json").write_text(json.dumps(report, indent=2, allow_nan=True) + "\n")
    print(json.dumps({k: report[k] for k in ("rows", "full_rank_rows", "label_counts", "best_mode", "checkpoint")}, indent=2, allow_nan=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
