#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load_json(path: str | None) -> dict:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    return json.loads(p.read_text())


def _metric_pair(stagea: dict, stageb: dict, key: str, higher_is_better: bool = True) -> dict:
    a = float(stagea.get(key, 0.0))
    b = float(stageb.get(key, 0.0))
    delta = a - b
    winner = "stageA" if (delta >= 0.0 if higher_is_better else delta <= 0.0) else "stageB"
    return {
        "stageA": a,
        "stageB": b,
        "delta_stageA_minus_stageB": delta,
        "higher_is_better": bool(higher_is_better),
        "winner": winner,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stagea_gate", required=True)
    ap.add_argument("--stageb_gate", required=True)
    ap.add_argument("--stagea_ckpt", default=None)
    ap.add_argument("--stageb_ckpt", default=None)
    ap.add_argument("--output_json", required=True)
    args = ap.parse_args()

    stagea = _load_json(args.stagea_gate)
    stageb = _load_json(args.stageb_gate)

    metrics = {
        "corrective_sign_mean_acc": _metric_pair(stagea, stageb, "corrective_sign_mean_acc", higher_is_better=True),
        "corrective_focus_mean_acc": _metric_pair(stagea, stageb, "corrective_focus_mean_acc", higher_is_better=True),
        "closeability_calibrated_balanced_acc": _metric_pair(stagea, stageb, "closeability_calibrated_balanced_acc", higher_is_better=True),
        "progress_balanced_acc": _metric_pair(stagea, stageb, "progress_balanced_acc", higher_is_better=True),
        "pair_calibrated_balanced_acc": _metric_pair(stagea, stageb, "pair_calibrated_balanced_acc", higher_is_better=True),
        "corrective_dyaw_aux_acc": _metric_pair(stagea, stageb, "corrective_dyaw_aux_acc", higher_is_better=True),
        "corrective_dyaw_residual_mae": _metric_pair(stagea, stageb, "corrective_dyaw_residual_mae", higher_is_better=False),
        "far_negative_ready_prob_mean": _metric_pair(stagea, stageb, "far_negative_ready_prob_mean", higher_is_better=False),
    }

    compare = {
        "decision": "stageA_main_candidate",
        "main_candidate": {
            "role": "stageA",
            "source": "canonical alignment main candidate",
            "ckpt": args.stagea_ckpt,
            "gate_report": args.stagea_gate,
        },
        "comparison_baseline": {
            "role": "stageB",
            "source": "progress-heavy comparison baseline",
            "ckpt": args.stageb_ckpt,
            "gate_report": args.stageb_gate,
        },
        "metrics": metrics,
        "takeaway": {
            "stageA_favored_for_corrective": metrics["corrective_sign_mean_acc"]["winner"] == "stageA",
            "stageA_favored_for_closeability": metrics["closeability_calibrated_balanced_acc"]["winner"] == "stageA",
            "stageB_favored_for_progress": metrics["progress_balanced_acc"]["winner"] == "stageB",
            "stageB_favored_for_pairwise": metrics["pair_calibrated_balanced_acc"]["winner"] == "stageB",
        },
    }

    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(compare, indent=2))
    print(json.dumps(compare, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
