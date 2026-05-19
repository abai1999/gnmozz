"""
offline_pose_field_probe.py

Offline probe for candidate-action scorer.
"""

import argparse
import json

import numpy as np
import torch

from prismatic.models.pose_field_scorer import PoseFieldScorer


def hist_dict(arr):
    uniq, cnt = np.unique(arr, return_counts=True)
    return {int(k): int(v) for k, v in zip(uniq, cnt)}


def masked_group_argmax(scores: torch.Tensor, candidate_group_index: torch.Tensor, group_index: torch.Tensor) -> torch.Tensor:
    mask = candidate_group_index.eq(group_index.unsqueeze(1))
    masked_scores = scores.masked_fill(~mask, -1e9)
    return masked_scores.argmax(dim=-1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_npz", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--output_json", type=str, required=True)
    args = parser.parse_args()

    data = np.load(args.dataset_npz)
    ckpt = torch.load(args.ckpt, map_location="cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = PoseFieldScorer(
        use_depth=ckpt.get("use_depth", True),
        use_base_action=ckpt.get("use_base_action", True),
        use_proprio=ckpt.get("use_proprio", True),
        use_target_context=ckpt.get("use_target_context", True),
        use_front_rgb=ckpt.get("use_front_rgb", False),
        use_wrist_rgb=ckpt.get("use_wrist_rgb", False),
        num_candidate_groups=int(ckpt.get("num_candidate_groups", 11)),
        fire_only_head=ckpt.get("fire_only_head", True),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()

    n_rows = int(data["wrist_depth"].shape[0])
    front_rgb = torch.from_numpy(
        (data["front_rgb"] if "front_rgb" in data else np.zeros((n_rows, 128, 128, 3), dtype=np.uint8))
        .astype(np.float32)
        .transpose(0, 3, 1, 2)
    ).to(device=device, dtype=torch.float32) / 255.0
    wrist_rgb = torch.from_numpy(
        (data["wrist_rgb"] if "wrist_rgb" in data else np.zeros((n_rows, 128, 128, 3), dtype=np.uint8))
        .astype(np.float32)
        .transpose(0, 3, 1, 2)
    ).to(device=device, dtype=torch.float32) / 255.0
    wd = torch.from_numpy(data["wrist_depth"]).to(device=device, dtype=torch.float32)
    pr = torch.from_numpy(data["proprio"]).to(device=device, dtype=torch.float32)
    ba = torch.from_numpy(data["base_action"]).to(device=device, dtype=torch.float32)
    gc = torch.from_numpy(data["gripper_context"]).to(device=device, dtype=torch.float32)
    si = torch.from_numpy(data["step_idx"]).to(device=device, dtype=torch.long)
    pid = torch.from_numpy(data["phase_id"]).to(device=device, dtype=torch.long)
    page = torch.from_numpy(data["phase_age"]).to(device=device, dtype=torch.float32)
    sr = torch.from_numpy(data["steps_since_last_replan"]).to(device=device, dtype=torch.float32)
    ca = torch.from_numpy(data["candidate_actions_local"]).to(device=device, dtype=torch.float32)
    cgi = torch.from_numpy(data["candidate_group_index"]).to(device=device, dtype=torch.long)
    improve = torch.from_numpy(data["candidate_improvement"]).to(device=device, dtype=torch.float32)
    oracle_score = torch.from_numpy(data["candidate_oracle_score"] if "candidate_oracle_score" in data else data["candidate_improvement"]).to(device=device, dtype=torch.float32)
    next_dist = torch.from_numpy(data["candidate_next_basin_distance"]).to(device=device, dtype=torch.float32)
    best = torch.from_numpy(data["best_candidate_index"]).to(device=device, dtype=torch.long)
    best_group = torch.from_numpy(data["best_group_index"]).to(device=device, dtype=torch.long)
    current_dist = torch.from_numpy(data["current_basin_distance"]).to(device=device, dtype=torch.float32)
    use_target_context = bool(getattr(model, "use_target_context", True))
    cur_delta = torch.from_numpy(data["current_delta_basin_target"]).to(device=device, dtype=torch.float32) if use_target_context else None
    dx_sign_t = torch.from_numpy(data["current_dx_sign"]).to(device=device, dtype=torch.long) if use_target_context else None
    dy_sign_t = torch.from_numpy(data["current_dy_sign"]).to(device=device, dtype=torch.long) if use_target_context else None
    dyaw_sign_t = torch.from_numpy(data["current_dyaw_sign"]).to(device=device, dtype=torch.long) if use_target_context else None
    basin_bin_t = torch.from_numpy(data["basin_distance_bin"]).to(device=device, dtype=torch.long) if use_target_context else None

    with torch.no_grad():
        outputs = model(
            front_rgb, wrist_rgb, wd, pr, ba, gc, si, ca,
            phase_id=pid, phase_age=page, steps_since_last_replan=sr,
            current_delta_basin_target=cur_delta,
            current_dx_sign=dx_sign_t,
            current_dy_sign=dy_sign_t,
            current_dyaw_sign=dyaw_sign_t,
            basin_distance_bin=basin_bin_t,
            return_aux=True,
        )
        scores = outputs["candidate_scores"]
        group_logits = outputs["group_logits"]
        ready_prob = outputs["ready_to_close"]
        pred_group = group_logits.argmax(dim=-1)
        pred = masked_group_argmax(scores, cgi, pred_group)
        group_probs = torch.softmax(group_logits, dim=-1)
        selected_group_prob = group_probs.gather(1, pred_group.unsqueeze(1)).squeeze(1)
        group_mask = cgi.eq(pred_group.unsqueeze(1))
        masked_scores = scores.masked_fill(~group_mask, -1e9)
        within_group_probs = torch.softmax(masked_scores, dim=-1)
        final_pred_prob = selected_group_prob * within_group_probs.gather(1, pred.unsqueeze(1)).squeeze(1)
        topk_prob, topk_idx = torch.topk(within_group_probs, k=min(3, scores.shape[1]), dim=-1)
        group_top2 = torch.topk(group_logits, k=min(2, group_logits.shape[1]), dim=-1).values
        group_margin = group_top2[:, 0] - group_top2[:, 1] if group_top2.shape[1] > 1 else torch.zeros_like(final_pred_prob)
        cand_top2 = torch.topk(masked_scores, k=min(2, masked_scores.shape[1]), dim=-1).values
        cand_margin = cand_top2[:, 0] - cand_top2[:, 1] if cand_top2.shape[1] > 1 else torch.zeros_like(final_pred_prob)
        final_logit_margin = group_margin + cand_margin

    top1 = float((pred == best).float().mean().item())
    group_acc = float((pred_group == best_group).float().mean().item())
    selected_improve = improve.gather(1, pred.unsqueeze(1)).squeeze(1)
    selected_oracle_score = oracle_score.gather(1, pred.unsqueeze(1)).squeeze(1)
    oracle_improve = oracle_score.gather(1, best.unsqueeze(1)).squeeze(1)
    selected_next = next_dist.gather(1, pred.unsqueeze(1)).squeeze(1)
    selected_action = ca.gather(1, pred[:, None, None].expand(-1, 1, ca.shape[-1])).squeeze(1)
    basin_bin = data["basin_distance_bin"].astype(np.int64)
    dx_sign = data["current_dx_sign"].astype(np.int64)
    dy_sign = data["current_dy_sign"].astype(np.int64)
    dyaw_sign = data["current_dyaw_sign"].astype(np.int64)
    planner_close_intent_np = data["planner_close_intent"].astype(np.float32) if "planner_close_intent" in data else np.zeros((ca.shape[0],), dtype=np.float32)
    ready_target_np = data["ready_to_close_target"].astype(np.float32) if "ready_to_close_target" in data else np.zeros((ca.shape[0],), dtype=np.float32)
    z_abs_np = np.abs(data["current_delta_basin_target"][:, 2]).astype(np.float32)
    geom_support_np = data["geometry_conditioned_pose_support"].astype(np.int64) if "geometry_conditioned_pose_support" in data else np.zeros((ca.shape[0],), dtype=np.int64)
    planner_support_np = data["planner_conditioned_support"].astype(np.int64) if "planner_conditioned_support" in data else np.zeros((ca.shape[0],), dtype=np.int64)
    background_support_np = data["background_align_support"].astype(np.int64) if "background_align_support" in data else np.zeros((ca.shape[0],), dtype=np.int64)

    pred_np = pred.cpu().numpy()
    ready_prob_np = ready_prob.cpu().numpy()
    ready_pred_np = (ready_prob_np >= 0.5).astype(np.float32)
    best_np = best.cpu().numpy()
    pred_prob_np = final_pred_prob.cpu().numpy()
    logit_margin_np = final_logit_margin.cpu().numpy()

    def split_metrics(mask):
        mask = np.asarray(mask, dtype=bool)
        if not np.any(mask):
            return {
                "count": 0,
                "top1_acc": 0.0,
                "candidate0_rate": 0.0,
                "nonzero_rate": 0.0,
                "top1_prob_mean": 0.0,
                "logit_margin_mean": 0.0,
                "pred_hist": {},
            }
        vals = pred_np[mask]
        return {
            "count": int(mask.sum()),
            "top1_acc": float(np.mean(vals == best_np[mask])),
            "candidate0_rate": float(np.mean(vals == 0)),
            "nonzero_rate": float(np.mean(vals != 0)),
            "top1_prob_mean": float(np.mean(pred_prob_np[mask])),
            "logit_margin_mean": float(np.mean(logit_margin_np[mask])),
            "pred_hist": hist_dict(vals),
        }

    high_z_no_close = (planner_close_intent_np <= 0.5) & (z_abs_np >= 0.025)
    mid_z_no_close = (planner_close_intent_np <= 0.5) & (z_abs_np >= 0.01) & (z_abs_np < 0.025)
    low_z_or_close = (planner_close_intent_np > 0.5) | (z_abs_np < 0.01)
    ready_tp = float(np.sum((ready_pred_np > 0.5) & (ready_target_np > 0.5)))
    ready_fp = float(np.sum((ready_pred_np > 0.5) & (ready_target_np <= 0.5)))
    ready_fn = float(np.sum((ready_pred_np <= 0.5) & (ready_target_np > 0.5)))
    ready_precision = ready_tp / max(ready_tp + ready_fp, 1.0)
    ready_recall = ready_tp / max(ready_tp + ready_fn, 1.0)
    ready_f1 = 0.0 if (ready_precision + ready_recall) <= 1e-8 else (2.0 * ready_precision * ready_recall / (ready_precision + ready_recall))

    grouped = {}
    for name, values in {
        "basin_distance_bin": basin_bin,
        "dx_sign": dx_sign,
        "dy_sign": dy_sign,
        "dyaw_sign": dyaw_sign,
    }.items():
        grouped[name] = {}
        for group_val in np.unique(values):
            mask = values == group_val
            grouped[name][int(group_val)] = {
                "count": int(mask.sum()),
                "oracle_best_hist": hist_dict(best.cpu().numpy()[mask]),
                "oracle_group_hist": hist_dict(best_group.cpu().numpy()[mask]),
                "pred_hist": hist_dict(pred.cpu().numpy()[mask]),
                "pred_group_hist": hist_dict(pred_group.cpu().numpy()[mask]),
                "selected_improvement_mean": float(selected_improve.cpu().numpy()[mask].mean()) if np.any(mask) else 0.0,
                "selected_oracle_score_mean": float(selected_oracle_score.cpu().numpy()[mask].mean()) if np.any(mask) else 0.0,
            }

    result = {
        "group_acc": group_acc,
        "top1_acc": top1,
        "selected_improvement_mean": float(selected_improve.mean().item()),
        "oracle_improvement_mean": float(oracle_improve.mean().item()),
        "selected_oracle_score_mean": float(selected_oracle_score.mean().item()),
        "selected_reduces_basin_rate": float((selected_improve > 0).float().mean().item()),
        "selected_next_basin_distance_mean": float(selected_next.mean().item()),
        "current_basin_distance_mean": float(current_dist.mean().item()),
        "selected_action_mean": selected_action.mean(dim=0).cpu().numpy().tolist(),
        "oracle_best_hist": hist_dict(best.cpu().numpy()),
        "oracle_group_hist": hist_dict(best_group.cpu().numpy()),
        "pred_hist": hist_dict(pred.cpu().numpy()),
        "pred_group_hist": hist_dict(pred_group.cpu().numpy()),
        "candidate_count": int(ca.shape[1]),
        "sample_count": int(ca.shape[0]),
        "final_top1_prob_mean": float(final_pred_prob.mean().item()),
        "final_logit_margin_mean": float(final_logit_margin.mean().item()),
        "candidate0_rate": float(np.mean(pred_np == 0)),
        "nonzero_correction_rate": float(np.mean(pred_np != 0)),
        "ready_acc": float(np.mean((ready_pred_np > 0.5) == (ready_target_np > 0.5))),
        "ready_precision": float(ready_precision),
        "ready_recall": float(ready_recall),
        "ready_f1": float(ready_f1),
        "ready_positive_rate": float(np.mean(ready_pred_np > 0.5)),
        "ready_target_positive_rate": float(np.mean(ready_target_np > 0.5)),
        "ready_prob_mean": float(np.mean(ready_prob_np)),
        "support_source_summary": {
            "geometry_conditioned_pose_support": int(np.sum(geom_support_np > 0)),
            "planner_conditioned_support": int(np.sum(planner_support_np > 0)),
            "background_align_support": int(np.sum(background_support_np > 0)),
        },
        "split_metrics": {
            "high_z_no_close": split_metrics(high_z_no_close),
            "mid_z_no_close": split_metrics(mid_z_no_close),
            "low_z_or_close": split_metrics(low_z_or_close),
        },
        "grouped_analysis": grouped,
    }
    if int(ca.shape[0]) <= 64:
        result["per_state_predictions"] = [
            {
                "row": int(i),
                "best_candidate_index": int(best[i].item()),
                "pred_candidate_index": int(pred[i].item()),
                "best_group_index": int(best_group[i].item()),
                "pred_group_index": int(pred_group[i].item()),
                "pred_group_prob": float(selected_group_prob[i].item()),
                "pred_final_prob": float(final_pred_prob[i].item()),
                "pred_final_logit_margin": float(final_logit_margin[i].item()),
                "topk_candidate_index": [int(x) for x in topk_idx[i].detach().cpu().numpy().tolist()],
                "topk_candidate_prob": [float(x) for x in topk_prob[i].detach().cpu().numpy().tolist()],
                "selected_oracle_score": float(selected_oracle_score[i].item()),
                "oracle_score": float(oracle_improve[i].item()),
            }
            for i in range(int(ca.shape[0]))
        ]
    with open(args.output_json, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
