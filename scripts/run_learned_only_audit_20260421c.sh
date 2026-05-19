#!/usr/bin/env bash
set -euo pipefail

LEARNED_NPZ="/home/guoning/code/VLA/runtime_artifacts/residual_data/insert_phase1_support_resync_learned_full32_20260421c/support_states.npz"
OUT_DIR="/home/guoning/code/VLA/runtime_artifacts/residual_data/audits/20260421c"
LOG_FILE="${OUT_DIR}/learned_only_audit_20260421c.runner.log"
OUT_JSON="${OUT_DIR}/learned_only_audit_20260421c.json"
OUT_MD="${OUT_DIR}/learned_only_audit_20260421c.md"

mkdir -p "${OUT_DIR}"
: > "${LOG_FILE}"

echo "[$(date '+%F %T')] waiting for learned32 completion..." >> "${LOG_FILE}"
while true; do
  running=0
  if pgrep -f "evaluate_rlbench_modes.py.*seed3407_full_learned32" >/dev/null; then
    running=1
  fi
  if [[ -f "${LEARNED_NPZ}" && "${running}" -eq 0 ]]; then
    break
  fi
  if [[ -f "${LEARNED_NPZ}" ]]; then
    size="$(stat -c%s "${LEARNED_NPZ}" 2>/dev/null || echo 0)"
    echo "[$(date '+%F %T')] learned npz present (${size}B), process still running=${running}" >> "${LOG_FILE}"
  else
    echo "[$(date '+%F %T')] learned npz missing, process running=${running}" >> "${LOG_FILE}"
  fi
  sleep 60
done

echo "[$(date '+%F %T')] learned32 finished; running learned-only audit..." >> "${LOG_FILE}"

/home/guoning/my_conda_envs/vla-adapter/bin/python - <<'PY'
import json
from pathlib import Path
import numpy as np

npz_path = Path("/home/guoning/code/VLA/runtime_artifacts/residual_data/insert_phase1_support_resync_learned_full32_20260421c/support_states.npz")
out_json = Path("/home/guoning/code/VLA/runtime_artifacts/residual_data/audits/20260421c/learned_only_audit_20260421c.json")
out_md = Path("/home/guoning/code/VLA/runtime_artifacts/residual_data/audits/20260421c/learned_only_audit_20260421c.md")

raw = np.load(npz_path, allow_pickle=False)

def arr(name):
    if name not in raw:
        raise KeyError(f"Missing key: {name}")
    return raw[name]

def finite_stats(x):
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {"count": 0, "mean": None, "median": None, "p95": None}
    return {
        "count": int(x.size),
        "mean": float(np.mean(x)),
        "median": float(np.median(x)),
        "p95": float(np.percentile(x, 95)),
    }

proxy = np.asarray(arr("proxy_current_delta_basin_target"), dtype=np.float64)
teacher = np.asarray(arr("teacher_current_delta_basin_target"), dtype=np.float64)
sep = np.linalg.norm(proxy - teacher, axis=1)
eq_ratio = float(np.mean(np.all(np.isclose(proxy, teacher, atol=1e-8, rtol=0.0), axis=1)))

xy = np.asarray(arr("teacher_truth_handoff_metric_xy_error"), dtype=np.float64)
z = np.abs(np.asarray(arr("teacher_truth_handoff_metric_abs_z_error"), dtype=np.float64))
yaw = np.asarray(arr("teacher_truth_handoff_metric_yaw_error"), dtype=np.float64)
rx = np.asarray(arr("teacher_truth_handoff_release_threshold_xy_error"), dtype=np.float64)
rz = np.asarray(arr("teacher_truth_handoff_release_threshold_abs_z_error"), dtype=np.float64)
ry = np.asarray(arr("teacher_truth_handoff_release_threshold_yaw_error"), dtype=np.float64)

phase = np.asarray(arr("phase_id"), dtype=np.int64)
gripper_open = np.asarray(arr("rollout_gripper_open"), dtype=np.float64) > 0.5
finite_mask = np.isfinite(xy) & np.isfinite(z) & np.isfinite(yaw) & np.isfinite(rx) & np.isfinite(rz) & np.isfinite(ry)
release_band = (xy <= rx) & (z <= rz) & (yaw <= ry)
near_ready = (
    finite_mask
    & (phase == 1)
    & gripper_open
    & (z <= 1.5 * rz)
    & (yaw <= 1.5 * ry)
    & (xy >= 1.0 * rx)
    & (xy <= 3.0 * rx)
    & (~release_band)
)

episode_index = np.asarray(arr("episode_index"), dtype=np.int64)
near_episode_cov = int(np.unique(episode_index[near_ready]).size) if np.any(near_ready) else 0

cand_scores = np.asarray(arr("candidate_oracle_score"), dtype=np.float64)
cand_groups = np.asarray(arr("candidate_group_index"), dtype=np.int64)
cand_mask = np.asarray(arr("candidate_mask"), dtype=np.float64) > 0.5
runtime_group = np.asarray(arr("runtime_selected_group_index"), dtype=np.int64)

overall_best = np.full((cand_scores.shape[0],), np.nan, dtype=np.float64)
reachable_best = np.full((cand_scores.shape[0],), np.nan, dtype=np.float64)
oracle_best_in_runtime_group = np.zeros((cand_scores.shape[0],), dtype=bool)

for i in range(cand_scores.shape[0]):
    valid = cand_mask[i]
    if not np.any(valid):
        continue
    s = cand_scores[i]
    overall_best[i] = float(np.max(s[valid]))
    best_idx = int(np.argmax(np.where(valid, s, -1e9)))
    rg = int(runtime_group[i])
    in_group = valid & (cand_groups[i] == rg)
    if np.any(in_group):
        reachable_best[i] = float(np.max(s[in_group]))
    oracle_best_in_runtime_group[i] = bool(cand_groups[i, best_idx] == rg)

regret_gap = overall_best - reachable_best
nr_valid = near_ready & np.isfinite(overall_best) & np.isfinite(reachable_best)

reachable_ratio = float(np.mean(oracle_best_in_runtime_group[nr_valid])) if np.any(nr_valid) else 0.0
mean_regret_gap = float(np.mean(regret_gap[nr_valid])) if np.any(nr_valid) else None
p95_regret_gap = float(np.percentile(regret_gap[nr_valid], 95)) if np.any(nr_valid) else None

group_probs = np.asarray(arr("group_probs"), dtype=np.float64)
top2 = np.partition(group_probs, -2, axis=1)[:, -2:]
p1 = np.maximum(top2[:, 1], 1e-9)
p2 = np.maximum(top2[:, 0], 1e-9)
group_margin = np.log(p1) - np.log(p2)
gm = group_margin[nr_valid] if np.any(nr_valid) else np.asarray([], dtype=np.float64)

near_rows = int(np.sum(near_ready))
sample_gate = bool((near_rows >= 200) or (near_episode_cov >= 15))

if not sample_gate:
    decision = "insufficient_sample_for_structure_decision"
elif reachable_ratio < 0.10:
    decision = "upgrade_to_near_ready_gated_group_logit_residual"
elif reachable_ratio < 0.25:
    decision = "hybrid_group_logit_residual_plus_intra_group_score_residual"
else:
    decision = "pure_score_residual_can_be_retested"

report = {
    "dataset": str(npz_path),
    "rows": int(cand_scores.shape[0]),
    "learned_only_main_table": {
        "near_ready_rows": near_rows,
        "near_ready_episode_coverage": near_episode_cov,
        "sample_gate_passed": sample_gate,
        "separation": {
            "sep_all_mean": float(np.mean(sep)),
            "sep_all_median": float(np.median(sep)),
            "sep_all_p95": float(np.percentile(sep, 95)),
            "eq_ratio": eq_ratio,
            "sep_near_ready": finite_stats(sep[near_ready]),
        },
        "reachability": {
            "reachable_ratio": reachable_ratio,
            "mean_regret_gap_overall_vs_reachable": mean_regret_gap,
            "p95_regret_gap_overall_vs_reachable": p95_regret_gap,
            "rows_used": int(np.sum(nr_valid)),
        },
        "group_margin": {
            "definition": "log(top1_prob)-log(top2_prob) from group_probs",
            "group_top1_top2_margin_mean": float(np.mean(gm)) if gm.size else None,
            "group_top1_top2_margin_p50": float(np.percentile(gm, 50)) if gm.size else None,
            "group_top1_top2_margin_lt_0_35_ratio": float(np.mean(gm < 0.35)) if gm.size else None,
        },
        "structure_decision_by_rule": decision,
    },
}

out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False))

tbl = report["learned_only_main_table"]
lines = [
    "# Learned-only Audit (Main Decision Table) - 20260421c",
    "",
    f"- dataset: `{npz_path}`",
    f"- rows: `{report['rows']}`",
    f"- near_ready_rows: `{tbl['near_ready_rows']}`",
    f"- near_ready_episode_coverage: `{tbl['near_ready_episode_coverage']}`",
    f"- sample_gate_passed: `{tbl['sample_gate_passed']}`",
    "",
    "## Separation",
    f"- sep_all mean / median / p95: `{tbl['separation']['sep_all_mean']:.6f}` / `{tbl['separation']['sep_all_median']:.6f}` / `{tbl['separation']['sep_all_p95']:.6f}`",
    f"- eq_ratio: `{tbl['separation']['eq_ratio']:.6f}`",
    "",
    "## Reachability",
    f"- reachable_ratio: `{tbl['reachability']['reachable_ratio']}`",
    f"- mean_regret_gap_overall_vs_reachable: `{tbl['reachability']['mean_regret_gap_overall_vs_reachable']}`",
    f"- p95_regret_gap_overall_vs_reachable: `{tbl['reachability']['p95_regret_gap_overall_vs_reachable']}`",
    f"- rows_used: `{tbl['reachability']['rows_used']}`",
    "",
    "## Group Margin",
    f"- group_top1_top2_margin_mean: `{tbl['group_margin']['group_top1_top2_margin_mean']}`",
    f"- group_top1_top2_margin_p50: `{tbl['group_margin']['group_top1_top2_margin_p50']}`",
    f"- group_top1_top2_margin_lt_0.35_ratio: `{tbl['group_margin']['group_top1_top2_margin_lt_0_35_ratio']}`",
    "",
    "## Decision",
    f"- structure_decision_by_rule: `{tbl['structure_decision_by_rule']}`",
]
out_md.write_text("\\n".join(lines))
print(out_json)
print(out_md)
PY

echo "[$(date '+%F %T')] learned-only audit completed." >> "${LOG_FILE}"
echo "[$(date '+%F %T')] json=${OUT_JSON}" >> "${LOG_FILE}"
echo "[$(date '+%F %T')] md=${OUT_MD}" >> "${LOG_FILE}"
