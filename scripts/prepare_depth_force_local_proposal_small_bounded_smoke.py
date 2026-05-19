#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage_report_json", required=True)
    ap.add_argument("--shadow_purity_json", required=True)
    ap.add_argument("--output_json", required=True)
    ap.add_argument("--top_k", type=int, default=3)
    ap.add_argument("--prefer_episodes", default="5,17,19,8,10,1,16,20")
    args = ap.parse_args()

    stage = _load_json(Path(args.stage_report_json))
    shadow = _load_json(Path(args.shadow_purity_json))
    episodes = stage.get("per_episode", {})
    shadow_eps = shadow.get("per_episode", {})
    prefer = [int(x) for x in str(args.prefer_episodes).split(",") if x.strip()]

    scored: list[tuple[float, int, dict]] = []
    for ep_str, rep in episodes.items():
        try:
            ep = int(ep_str)
        except Exception:
            continue
        if prefer and ep not in prefer:
            continue
        score = 0.0
        score += float(rep.get("selected_best_safe_hit_rate", 0.0)) * 2.0
        score += float(rep.get("selected_pareto_hit_rate", 0.0)) * 1.5
        score += float(rep.get("selected_geom_gain_mean", 0.0)) * 4.0
        score += max(0.0, 5.0 - float(rep.get("best_safe_rank_mean", 5.0))) * 0.3
        score += float(rep.get("selected_risk_delta_mean", 0.0)) * -2.0
        score += float(rep.get("fallback_rate", 0.0)) * -0.75
        score += float(shadow_eps.get(ep_str, {}).get("close_contact_rate", 0.0)) * 0.75
        scored.append((score, ep, rep))

    if not scored:
        raise RuntimeError("no episodes available for bounded smoke selection")

    scored.sort(key=lambda x: (-x[0], x[1]))
    picked = scored[: max(int(args.top_k), 1)]

    report = {
        "stage_report_json": str(args.stage_report_json),
        "shadow_purity_json": str(args.shadow_purity_json),
        "top_k": int(args.top_k),
        "prefer_episodes": prefer,
        "selected_episodes": [int(ep) for _, ep, _ in picked],
        "ranked_candidates": [
            {
                "episode": int(ep),
                "score": float(score),
                "selected_best_safe_hit_rate": float(rep.get("selected_best_safe_hit_rate", 0.0)),
                "selected_pareto_hit_rate": float(rep.get("selected_pareto_hit_rate", 0.0)),
                "selected_geom_gain_mean": float(rep.get("selected_geom_gain_mean", 0.0)),
                "best_safe_rank_mean": float(rep.get("best_safe_rank_mean", 0.0)),
                "fallback_rate": float(rep.get("fallback_rate", 0.0)),
                "close_contact_rate": float(shadow_eps.get(str(ep), {}).get("close_contact_rate", 0.0)),
            }
            for score, ep, rep in scored
        ],
    }
    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
