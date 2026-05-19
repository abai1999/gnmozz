#!/usr/bin/env python3
"""Runtime alignment causality audit: planner-only vs final_full comparison.

Covers: takeover timing, correction direction, template bias, alpha/clip,
near-zone gated replay, runtime input observability.
"""
from __future__ import annotations

import argparse, json, re
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np


def _ep_from_filename(path: Path) -> int:
    m = re.search(r"ep(\d+)", path.name)
    if m is None:
        raise ValueError(f"cannot extract episode from {path.name}")
    return int(m.group(1))


def _load_traces(trace_dir: Path) -> dict[int, list[dict]]:
    episodes: dict[int, list[dict]] = {}
    for path in sorted(trace_dir.glob("ep*_gripper_trace.jsonl")):
        ep = _ep_from_filename(path)
        rows: list[dict] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    r = json.loads(line)
                    r["_ep"] = ep
                    rows.append(r)
        episodes[ep] = sorted(rows, key=lambda r: int(r.get("step", 0)))
    return episodes


def _task_dir(run_dir: Path) -> Path:
    for p in sorted(run_dir.glob("*/eval_results.json")):
        return p.parent
    raise FileNotFoundError(f"no eval_results.json under {run_dir}")


# =========================================================================
# 1. PLANNER-ONLY vs FINAL_FULL COMPARISON
# =========================================================================

def _compare_handoff_metrics(ff_rows: list[dict], po_rows: list[dict]) -> dict:
    """Compare handoff metric trajectories between final_full and planner-only."""
    def _extract(rows):
        out = {"xy": [], "z": [], "yaw": [], "steps": []}
        for r in rows:
            hm = r.get("handoff_metrics_provider") or {}
            for key, store in [("xy_error", "xy"), ("abs_z_error", "z"), ("yaw_error", "yaw")]:
                v = hm.get(key)
                if v is not None and np.isfinite(float(v)):
                    out[store].append(float(v))
            out["steps"].append(r["step"])
        return out

    ff = _extract(ff_rows)
    po = _extract(po_rows)

    result = {}
    for axis in ["xy", "z", "yaw"]:
        ff_arr = np.array(ff[axis])
        po_arr = np.array(po[axis])
        # Compare at same-length segments
        n = min(len(ff_arr), len(po_arr))
        if n < 10:
            result[axis] = {"ff_mean": float(ff_arr.mean()), "po_mean": float(po_arr.mean()),
                            "delta": float(ff_arr.mean() - po_arr.mean()), "n": n}
            continue
        # Split into thirds
        third = n // 3
        ff_early = ff_arr[:third].mean()
        ff_mid = ff_arr[third:2*third].mean()
        ff_late = ff_arr[2*third:].mean()
        po_early = po_arr[:third].mean()
        po_mid = po_arr[third:2*third].mean()
        po_late = po_arr[2*third:].mean()
        ff_trend = ff_late - ff_early
        po_trend = po_late - po_early
        result[axis] = {
            "n": n,
            "ff_overall_mean": round(float(ff_arr.mean()), 6),
            "po_overall_mean": round(float(po_arr.mean()), 6),
            "ff_early_mean": round(float(ff_early), 6),
            "ff_mid_mean": round(float(ff_mid), 6),
            "ff_late_mean": round(float(ff_late), 6),
            "po_early_mean": round(float(po_early), 6),
            "po_mid_mean": round(float(po_mid), 6),
            "po_late_mean": round(float(po_late), 6),
            "ff_trend": round(float(ff_trend), 6),
            "po_trend": round(float(po_trend), 6),
            "final_full_better": bool(
                (axis == "z" and ff_late < po_late) or
                (axis in ("xy", "yaw") and ff_late < po_late)
            ) if n > 10 else None,
        }
    return result


# =========================================================================
# 2. TAKEOVER TIMING AUDIT
# =========================================================================

def _takeover_timing(rows: list[dict]) -> dict:
    first_tk = -1
    first_gate = -1
    for r in rows:
        if first_gate < 0 and r.get("refiner_alignment_gate_open"):
            first_gate = r["step"]
        if first_tk < 0 and r.get("refiner_alignment_takeover_active"):
            first_tk = r["step"]
            break

    takeover_rows = [r for r in rows if r.get("refiner_alignment_takeover_active")]
    ranges = [(0, 40), (40, 100), (100, 200), (200, 340)]
    tk_by_range = {}
    for lo, hi in ranges:
        tk_count = sum(1 for r in takeover_rows if lo <= r["step"] < hi)
        total = sum(1 for r in rows if lo <= r["step"] < hi)
        tk_by_range[f"steps_{lo}_{hi}"] = {
            "takeover_count": tk_count,
            "total_steps": total,
            "takeover_rate": round(tk_count / max(total, 1), 4),
        }

    # Error at first takeover
    first_tk_row = takeover_rows[0] if takeover_rows else None
    first_tk_error = None
    if first_tk_row:
        hm = first_tk_row.get("handoff_metrics_provider") or {}
        first_tk_error = {
            "xy": hm.get("xy_error"),
            "z": hm.get("abs_z_error"),
            "yaw": hm.get("yaw_error"),
        }

    # Support-to-takeover transition
    support_rows = [r for r in rows if r.get("refiner_alignment_blocked_reason") == "support"]
    last_support_step = support_rows[-1]["step"] if support_rows else -1

    return {
        "first_gate_open_step": first_gate,
        "first_takeover_step": first_tk,
        "last_support_blocked_step": last_support_step,
        "takeover_by_step_range": tk_by_range,
        "first_takeover_error": first_tk_error,
        "total_takeover_steps": len(takeover_rows),
        "total_steps": len(rows),
    }


# =========================================================================
# 3. CORRECTION DIRECTION AUDIT
# =========================================================================

def _correction_direction(rows: list[dict]) -> dict:
    """Analyze correction direction using handoff_aux residual predictions."""
    tk_rows = [r for r in rows if r.get("refiner_alignment_takeover_active")]

    dx_vals = []
    dy_vals = []
    dz_vals = []
    dyaw_vals = []
    xy_errors = []
    z_errors = []
    yaw_errors = []

    # Sign accuracy: does dx push toward reducing xy_error?
    dx_toward = 0
    dx_away = 0
    dy_toward = 0
    dy_away = 0
    dz_toward = 0
    dz_away = 0
    dyaw_toward = 0
    dyaw_away = 0

    for i, r in enumerate(tk_rows):
        ha = r.get("handoff_aux_provider") or {}
        hm = r.get("handoff_metrics_provider") or {}
        dx = ha.get("pred_residual_dx", 0) or 0
        dy = ha.get("pred_residual_dy", 0) or 0
        dz = ha.get("pred_residual_dz", 0) or 0
        dyaw = ha.get("pred_residual_dyaw", 0) or 0
        xy_e = hm.get("xy_error")
        z_e = hm.get("abs_z_error")
        yaw_e = hm.get("yaw_error")

        dx_vals.append(float(dx))
        dy_vals.append(float(dy))
        dz_vals.append(float(dz))
        dyaw_vals.append(float(dyaw))
        if xy_e is not None:
            xy_errors.append(float(xy_e))
        if z_e is not None:
            z_errors.append(float(z_e))
        if yaw_e is not None:
            yaw_errors.append(float(yaw_e))

        # Check next step to see if error improved
        if i + 1 < len(tk_rows):
            next_hm = tk_rows[i + 1].get("handoff_metrics_provider") or {}
            next_xy = next_hm.get("xy_error")
            next_z = next_hm.get("abs_z_error")
            next_yaw = next_hm.get("yaw_error")
            if xy_e is not None and next_xy is not None:
                if float(next_xy) < float(xy_e):
                    dx_toward += 1
                else:
                    dx_away += 1
            if z_e is not None and next_z is not None:
                if float(next_z) < float(z_e):
                    dz_toward += 1
                else:
                    dz_away += 1
            if yaw_e is not None and next_yaw is not None:
                if float(next_yaw) < float(yaw_e):
                    dyaw_toward += 1
                else:
                    dyaw_away += 1

    def _sign_stats(vals):
        a = np.array(vals)
        pos = int((a > 1e-8).sum())
        neg = int((a < -1e-8).sum())
        zero = int(len(a)) - pos - neg
        return {"positive": pos, "negative": neg, "zero": zero,
                "mean": round(float(a.mean()), 6), "std": round(float(a.std()), 6)}

    return {
        "n_takeover": len(tk_rows),
        "residual_dx": _sign_stats(dx_vals),
        "residual_dy": _sign_stats(dy_vals),
        "residual_dz": _sign_stats(dz_vals),
        "residual_dyaw": _sign_stats(dyaw_vals),
        "error_improvement": {
            "xy": {"improved": dx_toward, "worsened": dx_away,
                    "improve_rate": round(dx_toward / max(dx_toward + dx_away, 1), 4)},
            "z": {"improved": dz_toward, "worsened": dz_away,
                  "improve_rate": round(dz_toward / max(dz_toward + dz_away, 1), 4)},
            "yaw": {"improved": dyaw_toward, "worsened": dyaw_away,
                    "improve_rate": round(dyaw_toward / max(dyaw_toward + dyaw_away, 1), 4)},
        },
    }


# =========================================================================
# 4. TEMPLATE BIAS AUDIT
# =========================================================================

def _template_bias(rows: list[dict]) -> dict:
    tk_rows = [r for r in rows if r.get("refiner_alignment_takeover_active")]

    candidate_hist: dict[str, int] = {}
    for r in tk_rows:
        idx = r.get("refiner_last_scorer_candidate_index", -1)
        if idx is not None and idx >= 0:
            candidate_hist[str(int(idx))] = candidate_hist.get(str(int(idx)), 0) + 1

    # Residual sign distributions by step range
    ranges = [(0, 100), (100, 200), (200, 340)]
    range_stats = {}
    for lo, hi in ranges:
        rtk = [r for r in tk_rows if lo <= r["step"] < hi]
        dx_signs = []
        dy_signs = []
        dz_signs = []
        cands = Counter()
        for r in rtk:
            ha = r.get("handoff_aux_provider") or {}
            dx = float(ha.get("pred_residual_dx", 0) or 0)
            dy = float(ha.get("pred_residual_dy", 0) or 0)
            dz = float(ha.get("pred_residual_dz", 0) or 0)
            dx_signs.append(1 if dx > 1e-8 else (-1 if dx < -1e-8 else 0))
            dy_signs.append(1 if dy > 1e-8 else (-1 if dy < -1e-8 else 0))
            dz_signs.append(1 if dz > 1e-8 else (-1 if dz < -1e-8 else 0))
            idx = r.get("refiner_last_scorer_candidate_index", -1)
            if idx >= 0:
                cands[str(int(idx))] += 1
        dx_arr = np.array(dx_signs)
        dy_arr = np.array(dy_signs)
        dz_arr = np.array(dz_signs)
        range_stats[f"steps_{lo}_{hi}"] = {
            "n": len(rtk),
            "dx_sign_mean": round(float(dx_arr.mean()), 4) if dx_arr.size > 0 else 0,
            "dy_sign_mean": round(float(dy_arr.mean()), 4) if dy_arr.size > 0 else 0,
            "dz_sign_mean": round(float(dz_arr.mean()), 4) if dz_arr.size > 0 else 0,
            "dx_positive_rate": round(float((dx_arr > 0).sum() / max(dx_arr.size, 1)), 4),
            "dy_positive_rate": round(float((dy_arr > 0).sum() / max(dy_arr.size, 1)), 4),
            "dz_negative_rate": round(float((dz_arr < 0).sum() / max(dz_arr.size, 1)), 4),
            "top_candidates": dict(cands.most_common(3)),
        }

    all_dx = []
    all_dy = []
    all_dz = []
    for r in tk_rows:
        ha = r.get("handoff_aux_provider") or {}
        all_dx.append(float(ha.get("pred_residual_dx", 0) or 0))
        all_dy.append(float(ha.get("pred_residual_dy", 0) or 0))
        all_dz.append(float(ha.get("pred_residual_dz", 0) or 0))
    dx_a = np.array(all_dx)
    dy_a = np.array(all_dy)
    dz_a = np.array(all_dz)

    return {
        "candidate_histogram": candidate_hist,
        "dominant_candidate": max(candidate_hist, key=candidate_hist.get) if candidate_hist else "none",
        "candidate_concentration_top1": round(max(candidate_hist.values()) / max(len(tk_rows), 1), 4) if candidate_hist else 0,
        "residual_dx_overall_mean": round(float(dx_a.mean()), 6),
        "residual_dy_overall_mean": round(float(dy_a.mean()), 6),
        "residual_dz_overall_mean": round(float(dz_a.mean()), 6),
        "residual_dy_sign_bias": "positive_right" if dy_a.mean() > 0.0001 else ("negative_left" if dy_a.mean() < -0.0001 else "neutral"),
        "residual_dx_sign_bias": "positive" if dx_a.mean() > 0.0001 else ("negative" if dx_a.mean() < -0.0001 else "neutral"),
        "by_step_range": range_stats,
    }


# =========================================================================
# 5. NEAR-ZONE GATED REPLAY
# =========================================================================

def _near_zone_replay(rows: list[dict]) -> dict:
    """Simulate: only allow takeover if robot is close enough."""
    tk_rows = [r for r in rows if r.get("refiner_alignment_takeover_active")]
    total = len(tk_rows)

    gates = {
        "no_gate": {"cond": lambda r: True},
        "z_lt_0.10": {"cond": lambda r: _hm(r, "abs_z_error", 999) < 0.10},
        "z_lt_0.07": {"cond": lambda r: _hm(r, "abs_z_error", 999) < 0.07},
        "xy_lt_0.04": {"cond": lambda r: _hm(r, "xy_error", 999) < 0.04},
        "step_gt_100": {"cond": lambda r: r["step"] > 100},
        "step_gt_150": {"cond": lambda r: r["step"] > 150},
        "z_lt_0.10_and_xy_lt_0.05": {"cond": lambda r: _hm(r, "abs_z_error", 999) < 0.10 and _hm(r, "xy_error", 999) < 0.05},
        "z_lt_0.07_and_step_gt_100": {"cond": lambda r: _hm(r, "abs_z_error", 999) < 0.07 and r["step"] > 100},
    }

    results = {}
    for name, gate in gates.items():
        retained = [r for r in tk_rows if gate["cond"](r)]
        n = len(retained)
        # Metrics of retained steps
        xy_errs = [_hm(r, "xy_error") for r in retained if _hm(r, "xy_error") is not None]
        z_errs = [_hm(r, "abs_z_error") for r in retained if _hm(r, "abs_z_error") is not None]
        yaw_errs = [_hm(r, "yaw_error") for r in retained if _hm(r, "yaw_error") is not None]
        # Improvement rate
        n_improved = sum(1 for i, r in enumerate(retained)
                         if i + 1 < len(retained)
                         and _hm(retained[i+1], "abs_z_error") is not None
                         and _hm(r, "abs_z_error") is not None
                         and float(_hm(retained[i+1], "abs_z_error")) < float(_hm(r, "abs_z_error")))

        results[name] = {
            "retained": n, "retention_rate": round(n / max(total, 1), 4),
            "xy_error_mean": round(float(np.mean(xy_errs)), 6) if xy_errs else None,
            "z_error_mean": round(float(np.mean(z_errs)), 6) if z_errs else None,
            "yaw_error_mean": round(float(np.mean(yaw_errs)), 6) if yaw_errs else None,
            "z_improve_rate": round(n_improved / max(n - 1, 1), 4) if n > 1 else 0,
            "first_retained_step": retained[0]["step"] if retained else -1,
            "last_retained_step": retained[-1]["step"] if retained else -1,
        }

    return results


def _hm(r, key, default=None):
    hm = r.get("handoff_metrics_provider") or {}
    v = hm.get(key)
    if v is None or not np.isfinite(float(v)):
        return default
    return float(v)


# =========================================================================
# 6. ALPHA / CLIP AUDIT
# =========================================================================

def _alpha_clip_audit(rows: list[dict], eval_data: dict) -> dict:
    rs_list = eval_data.get("refiner_stats", [])
    ep_indices = eval_data.get("episode_indices", [])

    clip_by_ep = {}
    for i, rs in enumerate(rs_list):
        ep = ep_indices[i] if i < len(ep_indices) else i
        clip_by_ep[f"ep{ep:03d}"] = {
            "clip_hit_rate": float(rs.get("clip_hit_rate", -1)),
            "alpha_mean": float(rs.get("alpha_mean", -1)),
            "correction_count": int(rs.get("correction_count", -1)),
            "alignment_correction_count": int(rs.get("alignment_correction_count", -1)),
        }

    # Z trajectory around high-clip points
    tk_rows = [r for r in rows if r.get("refiner_alignment_takeover_active")]
    z_vals = np.array([_hm(r, "abs_z_error", np.nan) for r in tk_rows])
    z_vals = z_vals[np.isfinite(z_vals)]

    return {
        "by_episode": clip_by_ep,
        "note": "alpha=1.0 means no attenuation on clipped corrections",
        "z_trajectory": {
            "overall": {"n": len(z_vals), "mean": round(float(z_vals.mean()), 6),
                        "min": round(float(z_vals.min()), 6), "max": round(float(z_vals.max()), 6)},
            "first_50_pct": {"mean": round(float(z_vals[:len(z_vals)//2].mean()), 6)},
            "last_50_pct": {"mean": round(float(z_vals[len(z_vals)//2:].mean()), 6)},
        },
    }


# =========================================================================
# MAIN
# =========================================================================

def main():
    parser = argparse.ArgumentParser(description="Alignment causality audit")
    parser.add_argument("--final-full-run-dir", type=Path, required=True)
    parser.add_argument("--planner-only-run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    ff_task = _task_dir(args.final_full_run_dir)
    po_task = _task_dir(args.planner_only_run_dir)

    ff_traces = _load_traces(ff_task / "gripper_traces")
    po_traces = _load_traces(po_task / "gripper_traces")

    with (ff_task / "eval_results.json").open() as f:
        ff_eval = json.load(f)
    with (po_task / "eval_results.json").open() as f:
        po_eval = json.load(f)

    # Per-episode analysis
    per_ep = {}
    for ep in sorted(ff_traces.keys()):
        ff_rows = ff_traces[ep]
        po_rows = po_traces.get(ep, [])

        comparison = _compare_handoff_metrics(ff_rows, po_rows) if po_rows else None
        timing = _takeover_timing(ff_rows)
        corr_dir = _correction_direction(ff_rows)
        tmpl = _template_bias(ff_rows)
        near_zone = _near_zone_replay(ff_rows)

        per_ep[f"ep{ep:03d}"] = {
            "takeover_timing": timing,
            "correction_direction": corr_dir,
            "template_bias": tmpl,
            "near_zone_gated_replay": near_zone,
            "planner_only_comparison": comparison,
        }

    # Alpha/clip (aggregate)
    alpha_clip = _alpha_clip_audit(
        [r for rows in ff_traces.values() for r in rows], ff_eval
    )

    # Aggregate template bias across all episodes
    all_tk = [r for rows in ff_traces.values() for r in rows if r.get("refiner_alignment_takeover_active")]
    all_cand = Counter()
    all_dx = []
    all_dy = []
    for r in all_tk:
        idx = r.get("refiner_last_scorer_candidate_index", -1)
        if idx >= 0:
            all_cand[str(int(idx))] += 1
        ha = r.get("handoff_aux_provider") or {}
        all_dx.append(float(ha.get("pred_residual_dx", 0) or 0))
        all_dy.append(float(ha.get("pred_residual_dy", 0) or 0))

    dx_a = np.array(all_dx)
    dy_a = np.array(all_dy)

    # Runtime input observability
    observability = {
        "conclusion": "NO target-relative error signal in _run_depth_force_local_proposal inputs",
        "inputs_to_model_forward": [
            "front_rgb", "wrist_rgb", "wrist_depth", "force_history",
            "proprio (joint_positions + gripper_pose + gripper_open)",
            "planner_base_action_local", "stage_token", "contact_phase",
            "depth_proximity", "gripper_state"
        ],
        "not_included": [
            "motion_target_pose_7d",
            "target-relative xyz/yaw error (delta_basin_target)",
            "handoff geometry / handoff_target_pose_7d",
            "absolute end-effector pose in world/task frame"
        ],
        "candidate_local_features": [
            "Candidate-local depth stats (per-candidate depth penetration)",
            "Candidate-local force interactions (per-candidate force delta)",
            "cand_delta = proposal_action - planner_base_action (relative, not target-relative)"
        ],
        "implication": (
            "Model generates K=8 proposals from learned proposal_head(state), "
            "then scores them using state+depth+force+proposal features. "
            "Without target-relative error, the model cannot distinguish "
            "'proposal that moves toward target' from 'proposal that moves away'. "
            "This causes template bias — the model defaults to the most common "
            "correction pattern from training data, which may not align with "
            "the actual target direction in a given episode."
        ),
    }

    report = {
        "audit": "alignment_causality",
        "final_full_run_dir": str(args.final_full_run_dir),
        "planner_only_run_dir": str(args.planner_only_run_dir),
        "aggregate": {
            "total_takeover_steps_across_episodes": len(all_tk),
            "overall_candidate_histogram": dict(all_cand.most_common()),
            "overall_dx_sign": {
                "positive_rate": round(float((dx_a > 1e-8).sum() / max(len(dx_a), 1)), 4),
                "negative_rate": round(float((dx_a < -1e-8).sum() / max(len(dx_a), 1)), 4),
                "mean": round(float(dx_a.mean()), 6),
            },
            "overall_dy_sign": {
                "positive_rate": round(float((dy_a > 1e-8).sum() / max(len(dy_a), 1)), 4),
                "negative_rate": round(float((dy_a < -1e-8).sum() / max(len(dy_a), 1)), 4),
                "mean": round(float(dy_a.mean()), 6),
                "interpretation": "positive dy = pushing right (viewed from robot)",
            },
            "alpha_clip_audit": alpha_clip,
            "runtime_input_observability": observability,
        },
        "per_episode": per_ep,
    }

    output_path = args.output or (ff_task / "alignment_causality_audit.json")
    with output_path.open("w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)

    # Print summary
    print(f"[causality] report -> {output_path}")
    print(f"\n=== PLANNER-ONLY vs FINAL_FULL COMPARISON ===")
    for ep_label, d in per_ep.items():
        comp = d["planner_only_comparison"]
        if comp:
            for axis in ["xy", "z", "yaw"]:
                c = comp[axis]
                if "ff_late_mean" in c:
                    ff_better = "FF" if c.get("final_full_better") else "PO"
                    print(f"  {ep_label} {axis}: FF_late={c['ff_late_mean']:.4f} PO_late={c['po_late_mean']:.4f} -> {ff_better} better")
        tmpl = d["template_bias"]
        print(f"  {ep_label} template: dominant_candidate={tmpl['dominant_candidate']} "
              f"concentration={tmpl['candidate_concentration_top1']} "
              f"dy_bias={tmpl['residual_dy_sign_bias']} "
              f"dy_mean={tmpl['residual_dy_overall_mean']:.6f}")

    print(f"\n=== TAKEOVER TIMING ===")
    for ep_label, d in per_ep.items():
        tt = d["takeover_timing"]
        print(f"  {ep_label}: first_gate={tt['first_gate_open_step']} first_tk={tt['first_takeover_step']} "
              f"last_support={tt['last_support_blocked_step']}")
        for rng, stats in tt["takeover_by_step_range"].items():
            print(f"    {rng}: tk={stats['takeover_count']}/{stats['total_steps']} rate={stats['takeover_rate']}")
        fte = tt.get("first_takeover_error", {}) or {}
        print(f"    first_tk_error: xy={fte.get('xy')} z={fte.get('z')} yaw={fte.get('yaw')}")

    print(f"\n=== CORRECTION DIRECTION ===")
    for ep_label, d in per_ep.items():
        cd = d["correction_direction"]
        print(f"  {ep_label}:")
        for axis in ["residual_dx", "residual_dy", "residual_dz", "residual_dyaw"]:
            s = cd[axis]
            print(f"    {axis}: pos={s['positive']} neg={s['negative']} zero={s['zero']} mean={s['mean']}")
        ei = cd["error_improvement"]
        for axis in ["xy", "z", "yaw"]:
            print(f"    {axis}_improve: {ei[axis]['improved']}/{ei[axis]['improved']+ei[axis]['worsened']} rate={ei[axis]['improve_rate']}")

    print(f"\n=== NEAR-ZONE GATED REPLAY (ep005) ===")
    for ep_label, d in per_ep.items():
        if ep_label == "ep005":
            nz = d["near_zone_gated_replay"]
            for gate, stats in nz.items():
                print(f"  {gate}: retained={stats['retained']} ({stats['retention_rate']}) z_mean={stats['z_error_mean']} z_improve={stats['z_improve_rate']}")

    print(f"\n=== RUNTIME INPUT OBSERVABILITY ===")
    print(f"  {observability['conclusion']}")
    for inp in observability["not_included"]:
        print(f"    MISSING: {inp}")
    print(f"  Implication: {observability['implication'][:200]}...")


if __name__ == "__main__":
    main()
