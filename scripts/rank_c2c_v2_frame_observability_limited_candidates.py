#!/usr/bin/env python3
"""Rank frame-observability-limited C2C v2 rows by calibrated yaw estimator score."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from prismatic.robot.coarse2contact_v2.frame_yaw_estimator import (
    frame_yaw_feature_vector,
    load_frame_yaw_checkpoint,
    resolve_yaw_observable_threshold,
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except Exception:
        return float(default)
    return out if np.isfinite(out) else float(default)


def _row_mapping(row: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = row.get(key)
    return value if isinstance(value, Mapping) else {}


def _is_frame_observability_limited(row: Mapping[str, Any]) -> bool:
    primary = str(row.get("yaw_observability_primary_blocker", ""))
    combo = str(row.get("yaw_observability_blocker_combo", ""))
    return bool(primary == "frame_observability_lt_010" or "frame_observability_lt_010" in combo)


def _episode_bucket_key(row: Mapping[str, Any]) -> tuple[int, str]:
    return int(row.get("episode_idx", -1)), str(row.get("failure_bucket", ""))


def _group_counts(rows: Iterable[Mapping[str, Any]], key_fn) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for row in rows:
        counter[str(key_fn(row))] += 1
    return dict(sorted(counter.items(), key=lambda item: item[0]))


def rank_candidates(
    rows: list[dict[str, Any]],
    *,
    checkpoint: Path,
    threshold: float | None = None,
) -> dict[str, Any]:
    model, metadata = load_frame_yaw_checkpoint(checkpoint, map_location="cpu")
    resolved_threshold = float(threshold) if threshold is not None else resolve_yaw_observable_threshold(metadata, default=0.5)

    candidates = [row for row in rows if _is_frame_observability_limited(row)]
    if not candidates:
        return {
            "schema_version": "frame_observability_limited_rank_v1",
            "checkpoint": str(checkpoint.resolve()),
            "threshold": float(resolved_threshold),
            "candidate_rows": 0,
            "recoverable_rows": 0,
            "candidate_episode_counts": {},
            "candidate_failure_bucket_counts": {},
            "recoverable_episode_counts": {},
            "recoverable_failure_bucket_counts": {},
            "rows": [],
        }

    features = np.stack([frame_yaw_feature_vector(row) for row in candidates]).astype(np.float32)
    with torch.no_grad():
        out = model(torch.as_tensor(features, dtype=torch.float32))
    probs = out["yaw_observable_probability"].detach().cpu().numpy().astype(np.float32)

    ranked_rows: list[dict[str, Any]] = []
    for idx, (row, prob) in enumerate(zip(candidates, probs)):
        prob = float(prob)
        recoverable = prob >= float(resolved_threshold)
        summary = {
            "rank_all": int(idx),
            "episode_idx": int(row.get("episode_idx", -1)),
            "step_idx": int(row.get("step_idx", row.get("step", -1))),
            "stage_name": str(row.get("stage_name", "")),
            "skill_type": str(row.get("skill_type", "")),
            "failure_bucket": str(row.get("failure_bucket", "")),
            "visual_observability_class": str(row.get("visual_observability_class", "")),
            "yaw_observability_class": str(row.get("yaw_observability_class", "")),
            "yaw_observability_primary_blocker": str(row.get("yaw_observability_primary_blocker", "")),
            "yaw_observability_blocker_combo": str(row.get("yaw_observability_blocker_combo", "")),
            "xy_error": _safe_float(row.get("xy_error"), float("nan")),
            "yaw_abs": _safe_float(row.get("yaw_abs"), float("nan")),
            "true_dyaw": _safe_float(_row_mapping(row, "true_basin_error_t").get("dyaw", row.get("privileged_dyaw", float("nan"))), float("nan")),
            "proxy_image_axis_yaw": _safe_float(_row_mapping(row, "proxy_local_geometry_error").get("image_axis_yaw", row.get("proxy_image_axis_yaw", float("nan"))), float("nan")),
            "proxy_residual_yaw": _safe_float(_row_mapping(row, "proxy_local_geometry_error").get("dyaw", row.get("proxy_residual_yaw", float("nan"))), float("nan")),
            "estimator_yaw_observable_probability": prob,
            "estimator_yaw_observable_threshold": float(resolved_threshold),
            "estimator_yaw_observable": bool(recoverable),
            "estimator_yaw_margin": float(prob - float(resolved_threshold)),
            "recoverable_by_estimator": bool(recoverable),
        }
        ranked_rows.append(summary)

    ranked_rows.sort(
        key=lambda row: (
            int(row["episode_idx"]),
            str(row["failure_bucket"]),
            -float(row["estimator_yaw_observable_probability"]),
            int(row["step_idx"]),
        )
    )

    for rank, row in enumerate(ranked_rows, start=1):
        row["rank_in_report"] = int(rank)

    recoverable_rows = [row for row in ranked_rows if bool(row["recoverable_by_estimator"])]

    candidate_episode_counts = _group_counts(ranked_rows, lambda r: r["episode_idx"])
    candidate_failure_bucket_counts = _group_counts(ranked_rows, lambda r: r["failure_bucket"])
    recoverable_episode_counts = _group_counts(recoverable_rows, lambda r: r["episode_idx"])
    recoverable_failure_bucket_counts = _group_counts(recoverable_rows, lambda r: r["failure_bucket"])

    by_episode: list[dict[str, Any]] = []
    episode_groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in ranked_rows:
        episode_groups[int(row["episode_idx"])].append(row)
    for episode_idx in sorted(episode_groups):
        subset = episode_groups[episode_idx]
        recovered = [row for row in subset if row["recoverable_by_estimator"]]
        by_episode.append(
            {
                "episode_idx": int(episode_idx),
                "rows": int(len(subset)),
                "recoverable_rows": int(len(recovered)),
                "recoverable_rate": float(len(recovered) / len(subset)) if subset else 0.0,
                "top_probability": float(max(row["estimator_yaw_observable_probability"] for row in subset)) if subset else 0.0,
                "median_probability": float(np.median([row["estimator_yaw_observable_probability"] for row in subset])) if subset else 0.0,
            }
        )

    by_failure_bucket: list[dict[str, Any]] = []
    bucket_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in ranked_rows:
        bucket_groups[str(row["failure_bucket"])].append(row)
    for bucket in sorted(bucket_groups):
        subset = bucket_groups[bucket]
        recovered = [row for row in subset if row["recoverable_by_estimator"]]
        by_failure_bucket.append(
            {
                "failure_bucket": bucket,
                "rows": int(len(subset)),
                "recoverable_rows": int(len(recovered)),
                "recoverable_rate": float(len(recovered) / len(subset)) if subset else 0.0,
                "top_probability": float(max(row["estimator_yaw_observable_probability"] for row in subset)) if subset else 0.0,
                "median_probability": float(np.median([row["estimator_yaw_observable_probability"] for row in subset])) if subset else 0.0,
            }
        )

    report = {
        "schema_version": "frame_observability_limited_rank_v1",
        "checkpoint": str(checkpoint.resolve()),
        "threshold": float(resolved_threshold),
        "candidate_rows": int(len(ranked_rows)),
        "recoverable_rows": int(len(recoverable_rows)),
        "recoverable_rate": float(len(recoverable_rows) / len(ranked_rows)) if ranked_rows else 0.0,
        "candidate_episode_counts": candidate_episode_counts,
        "candidate_failure_bucket_counts": candidate_failure_bucket_counts,
        "recoverable_episode_counts": recoverable_episode_counts,
        "recoverable_failure_bucket_counts": recoverable_failure_bucket_counts,
        "by_episode": by_episode,
        "by_failure_bucket": by_failure_bucket,
        "top_recoverable_rows": recoverable_rows[:50],
        "top_rows": ranked_rows[:50],
        "rows": ranked_rows,
    }
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description="Rank frame-observability-limited rows by calibrated yaw estimator probability.")
    ap.add_argument("--relabel_jsonl", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, default=Path("runtime_artifacts/coarse2contact_v2/checkpoints/frame_yaw_estimator_observability_balanced.pt"))
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--output_dir", type=Path, default=Path("runtime_artifacts/coarse2contact_v2/reports/frame_observability_limited_ranking"))
    args = ap.parse_args()

    rows = _read_jsonl(args.relabel_jsonl)
    report = rank_candidates(rows, checkpoint=args.checkpoint, threshold=args.threshold)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    out_json = output_dir / "frame_observability_limited_ranking.json"
    out_jsonl = output_dir / "frame_observability_limited_ranking.jsonl"
    out_md = output_dir / "frame_observability_limited_ranking.md"
    out_json.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    with open(out_jsonl, "w", encoding="utf-8") as handle:
        for row in report["rows"]:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    md = [
        "# Frame Observability Limited Ranking",
        "",
        f"- source: `{args.relabel_jsonl}`",
        f"- checkpoint: `{report['checkpoint']}`",
        f"- threshold: `{report['threshold']:.6f}`",
        f"- candidate_rows: `{report['candidate_rows']}`",
        f"- recoverable_rows: `{report['recoverable_rows']}`",
        f"- recoverable_rate: `{report['recoverable_rate']:.3f}`",
        "",
        "## By Episode",
    ]
    for item in report["by_episode"]:
        md.append(
            f"- ep{int(item['episode_idx']):03d}: rows={item['rows']}, recoverable={item['recoverable_rows']}, "
            f"rate={item['recoverable_rate']:.3f}, top_prob={item['top_probability']:.3f}"
        )
    md.append("")
    md.append("## By Failure Bucket")
    for item in report["by_failure_bucket"]:
        md.append(
            f"- `{item['failure_bucket']}`: rows={item['rows']}, recoverable={item['recoverable_rows']}, "
            f"rate={item['recoverable_rate']:.3f}, top_prob={item['top_probability']:.3f}"
        )
    md.append("")
    md.append("## Top Recoverable")
    for row in report["top_recoverable_rows"][:20]:
        md.append(
            f"- ep{int(row['episode_idx']):03d} step{int(row['step_idx'])}: bucket={row['failure_bucket']}, "
            f"prob={row['estimator_yaw_observable_probability']:.4f}, margin={row['estimator_yaw_margin']:.4f}, "
            f"xy={row['xy_error']:.4f}, yaw={row['yaw_abs']:.4f}"
        )
    out_md.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ["candidate_rows", "recoverable_rows", "recoverable_rate", "threshold"]}, indent=2, sort_keys=True))
    print(out_json)
    print(out_jsonl)
    print(out_md)


if __name__ == "__main__":
    main()
