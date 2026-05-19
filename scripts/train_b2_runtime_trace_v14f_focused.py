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


def iter_rows(trace_dir: Path):
    for path in find_trace_files(trace_dir):
        ep = episode_from_path(path)
        with path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                row["_episode_index"] = int(ep)
                row["_trace_file"] = path.name
                yield row


def as_bool(row: dict, key: str) -> bool:
    return bool(row.get(key, False))


def as_float(row: dict, key: str, default: float = math.nan) -> float:
    try:
        value = float(row.get(key, default))
    except Exception:
        return default
    return value if math.isfinite(value) else default


def select_rows(trace_dir: Path) -> tuple[list[dict], dict]:
    selected = []
    reasons = {
        "gate_open": 0,
        "negative_regret": 0,
        "close_intent": 0,
        "xy_or_z_block": 0,
        "apply_or_changed": 0,
        "final_selected": 0,
    }
    for row in iter_rows(trace_dir):
        if not as_bool(row, "b2_candidate_shadow_gate_open"):
            continue
        reasons["gate_open"] += 1
        regret = as_float(row, "b2_candidate_shadow_regret_delta", math.nan)
        if not math.isfinite(regret) or regret >= 0.0:
            continue
        reasons["negative_regret"] += 1
        if not as_bool(row, "refiner_alignment_planner_close_intent"):
            continue
        reasons["close_intent"] += 1
        blocked = str(row.get("refiner_close_blocked_reason", ""))
        if blocked not in ("xy", "z"):
            continue
        reasons["xy_or_z_block"] += 1
        if not (
            as_bool(row, "b2_candidate_shadow_changed")
            or str(row.get("b2_candidate_shadow_mode", "")).lower() == "apply"
        ):
            continue
        reasons["apply_or_changed"] += 1
        selected.append(row)
    reasons["final_selected"] = len(selected)
    return selected, reasons


def make_dataset(rows: list[dict]) -> dict[str, np.ndarray]:
    if not rows:
        raise RuntimeError("no focused v14f rows selected")
    deltas = []
    actions = []
    mask = []
    scope = []
    labels = []
    conf = []
    episodes = []
    steps = []
    for row in rows:
        delta = np.asarray(
            row.get("refiner_current_delta_basin_target", row.get("current_delta_basin_target", [0, 0, 0, 0, 0, 0])),
            dtype=np.float32,
        ).reshape(-1)
        act = np.asarray(row.get("b2_candidate_shadow_candidate_actions_local", []), dtype=np.float32)
        if delta.size < 6 or act.ndim != 2 or act.shape[1] < 6:
            continue
        act = act[:, :6].astype(np.float32)
        cand_mask = np.asarray(row.get("b2_candidate_shadow_candidate_valid_mask", np.ones((act.shape[0],), dtype=np.float32)), dtype=np.float32).reshape(-1)
        cand_scope = np.asarray(row.get("b2_candidate_shadow_candidate_scope_mask", np.ones((act.shape[0],), dtype=np.float32)), dtype=np.float32).reshape(-1)
        if cand_mask.shape[0] != act.shape[0] or cand_scope.shape[0] != act.shape[0]:
            continue
        deltas.append(delta[:6].astype(np.float32))
        actions.append(act)
        mask.append(cand_mask.astype(np.float32))
        scope.append(cand_scope.astype(np.float32))
        labels.append(0)
        conf.append(float(np.clip(1.0 + abs(as_float(row, "b2_candidate_shadow_regret_delta", -0.1)), 1.0, 4.0)))
        episodes.append(int(row.get("_episode_index", -1)))
        steps.append(int(row.get("step", -1)))
    if not deltas:
        raise RuntimeError("focused v14f selection left no usable rows")
    lengths = {x.shape[0] for x in actions}
    if len(lengths) != 1:
        raise RuntimeError(f"variable candidate lengths not supported: {sorted(lengths)}")
    return {
        "proxy_current_delta_basin_target": np.stack(deltas, axis=0).astype(np.float32),
        "candidate_actions_local": np.stack(actions, axis=0).astype(np.float32),
        "candidate_mask": np.stack(mask, axis=0).astype(np.float32),
        "b2_yaw_aware_candidate_scope_v3": np.stack(scope, axis=0).astype(np.float32),
        "yaw_mode3_label_v11": np.asarray(labels, dtype=np.int64),
        "yaw_mode_valid_v11": np.ones((len(labels),), dtype=np.float32),
        "yaw_mode_confidence_v11": np.asarray(conf, dtype=np.float32),
        "episode_index": np.asarray(episodes, dtype=np.int64),
        "step_idx": np.asarray(steps, dtype=np.int64),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace_dir", type=Path, required=True)
    ap.add_argument("--base_ckpt", type=Path, required=True)
    ap.add_argument("--output_dir", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=3407)
    args = ap.parse_args()

    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows, selection_report = select_rows(args.trace_dir)
    data = make_dataset(rows)
    np.savez_compressed(args.output_dir / "b2_runtime_trace_v14f_focused_dataset.npz", **data)

    ckpt = torch.load(args.base_ckpt, map_location="cpu")
    model = StudentCandidateEvaluatorV2(yaw_mode_classes=int(ckpt.get("yaw_mode_num_classes", 3)))
    model.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=False)
    model.set_mode_input_path("summary_only")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for p in model.parameters():
        p.requires_grad = False
    for name in ("delta_encoder", "candidate_summary_encoder", "summary_context_head", "yaw_mode_head"):
        for p in getattr(model, name).parameters():
            p.requires_grad = True
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=float(args.lr), weight_decay=1e-4)

    x_delta = torch.from_numpy(data["proxy_current_delta_basin_target"]).to(device)
    x_actions = torch.from_numpy(data["candidate_actions_local"]).to(device)
    x_mask = torch.from_numpy(data["candidate_mask"]).to(device)
    x_scope = torch.from_numpy(data["b2_yaw_aware_candidate_scope_v3"]).to(device)
    y = torch.from_numpy(data["yaw_mode3_label_v11"]).long().to(device)
    conf = torch.from_numpy(data["yaw_mode_confidence_v11"]).float().to(device)
    latent = torch.zeros((x_delta.shape[0], 128), device=device)

    order = np.arange(x_delta.shape[0], dtype=np.int64)
    history = []
    for epoch in range(1, int(args.epochs) + 1):
        np.random.shuffle(order)
        losses = []
        model.train()
        for start in range(0, order.size, int(args.batch_size)):
            idx = torch.from_numpy(order[start : start + int(args.batch_size)]).long().to(device)
            out = model.forward_with_mode(
                handoff_latent=latent[idx],
                proxy_current_delta_basin_target=x_delta[idx],
                candidate_actions_local=x_actions[idx],
                candidate_mask=x_mask[idx],
                yaw_aware_candidate_scope=x_scope[idx],
            )
            loss_vec = F.cross_entropy(out["yaw_mode_logits"], y[idx], reduction="none")
            weights = torch.clamp(conf[idx], 1.0, 4.0)
            loss = (loss_vec * weights).sum() / torch.clamp(weights.sum(), min=1e-6)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))
        with torch.no_grad():
            model.eval()
            out = model.forward_with_mode(
                handoff_latent=latent,
                proxy_current_delta_basin_target=x_delta,
                candidate_actions_local=x_actions,
                candidate_mask=x_mask,
                yaw_aware_candidate_scope=x_scope,
            )
            pred = torch.argmax(out["yaw_mode_logits"], dim=-1)
            keep_rate = float(torch.mean((pred == 0).float()).item())
            history.append({"epoch": epoch, "loss": float(np.mean(losses)) if losses else math.nan, "pred_keep_rate": keep_rate})

    out_ckpt = dict(ckpt)
    out_ckpt["model_state_dict"] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    out_ckpt["mode_input_path"] = "summary_only"
    out_ckpt["mode_feature_version"] = "runtime_trace_v14f_focused_negative_only"
    out_ckpt["runtime_trace_v14f"] = {
        "trace_dir": str(args.trace_dir),
        "selection_report": selection_report,
        "rows": int(x_delta.shape[0]),
        "history": history,
    }
    ckpt_path = args.output_dir / "student_candidate_evaluator_v2_v14f_focused.pt"
    torch.save(out_ckpt, ckpt_path)

    report = {
        "trace_dir": str(args.trace_dir),
        "dataset_npz": str(args.output_dir / "b2_runtime_trace_v14f_focused_dataset.npz"),
        "checkpoint": str(ckpt_path),
        "selection_report": selection_report,
        "rows": int(x_delta.shape[0]),
        "episodes": sorted(int(x) for x in np.unique(data["episode_index"]).tolist()),
        "history": history,
    }
    (args.output_dir / "runtime_trace_v14f_focused_report.json").write_text(json.dumps(report, indent=2, allow_nan=True) + "\n")
    print(json.dumps(report, indent=2, allow_nan=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
