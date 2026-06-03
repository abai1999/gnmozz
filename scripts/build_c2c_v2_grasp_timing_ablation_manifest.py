#!/usr/bin/env python3
"""Build timing-ablation candidate manifests for hard-bucket large XY support."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from scripts.c2c_v2_grasp_probe_metrics import safe_int


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True) + "\n")


def _row_key(row: Mapping[str, Any]) -> tuple[int, int]:
    return (
        safe_int(row.get("episode_idx", -1)),
        safe_int(row.get("step_idx", row.get("step", -1))),
    )


def _active_keys(rows: Iterable[Mapping[str, Any]]) -> set[tuple[int, int]]:
    keys: set[tuple[int, int]] = set()
    for row in rows:
        if bool(row.get("intervention_active", row.get("grasp_probe_active", False))):
            keys.add(_row_key(row))
    return keys


def _filter_candidates(
    candidates: list[dict[str, Any]],
    *,
    selected_keys: set[tuple[int, int]],
    failure_bucket: str | None,
    episodes: set[int] | None = None,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in candidates:
        if failure_bucket and str(row.get("failure_bucket", "")) != failure_bucket:
            continue
        if episodes and safe_int(row.get("episode_idx", -1)) not in episodes:
            continue
        if _row_key(row) not in selected_keys:
            continue
        output.append(dict(row))
    output.sort(key=lambda row: (_row_key(row)[0], _row_key(row)[1]))
    return output


def build_timing_ablation_manifests(
    base_candidates: list[dict[str, Any]],
    v16_active_rows: Iterable[Mapping[str, Any]],
    v17_active_rows: Iterable[Mapping[str, Any]],
    *,
    failure_bucket: str,
    episodes: set[int],
) -> dict[str, list[dict[str, Any]]]:
    v16_keys = _active_keys(v16_active_rows)
    v17_keys = _active_keys(v17_active_rows)
    return {
        "v16_active_set_only": _filter_candidates(
            base_candidates,
            selected_keys=v16_keys,
            failure_bucket=failure_bucket,
            episodes=episodes,
        ),
        "v17_new_active_set_only": _filter_candidates(
            base_candidates,
            selected_keys=v17_keys - v16_keys,
            failure_bucket=failure_bucket,
            episodes=episodes,
        ),
        "combined": _filter_candidates(
            base_candidates,
            selected_keys=v16_keys | v17_keys,
            failure_bucket=failure_bucket,
            episodes=episodes,
        ),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Build timing-ablation manifests from active-set audit rows.")
    ap.add_argument("--base_candidate_jsonl", type=Path, required=True)
    ap.add_argument("--v16_active_rows_jsonl", type=Path, action="append", required=True)
    ap.add_argument("--v17_active_rows_jsonl", type=Path, action="append", required=True)
    ap.add_argument("--failure_bucket", type=str, default="large_xy_large_yaw")
    ap.add_argument("--episodes", type=str, default="20,21")
    ap.add_argument("--output_dir", type=Path, required=True)
    args = ap.parse_args()

    base_candidates = _read_jsonl(args.base_candidate_jsonl)
    v16_rows = [row for path in args.v16_active_rows_jsonl for row in _read_jsonl(path)]
    v17_rows = [row for path in args.v17_active_rows_jsonl for row in _read_jsonl(path)]
    episodes = {safe_int(part) for part in str(args.episodes).split(",") if str(part).strip()}
    manifests = build_timing_ablation_manifests(
        base_candidates,
        v16_rows,
        v17_rows,
        failure_bucket=str(args.failure_bucket),
        episodes=episodes,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {
        "base_candidate_jsonl": str(args.base_candidate_jsonl.resolve()),
        "v16_active_rows_jsonl": [str(path.resolve()) for path in args.v16_active_rows_jsonl],
        "v17_active_rows_jsonl": [str(path.resolve()) for path in args.v17_active_rows_jsonl],
        "failure_bucket": str(args.failure_bucket),
        "episodes": sorted(int(ep) for ep in episodes),
        "counts": {mode: len(rows) for mode, rows in manifests.items()},
    }
    (args.output_dir / "timing_ablation_manifest_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    for mode, rows in manifests.items():
        _write_jsonl(args.output_dir / f"grasp_failure_tail_candidates_{str(args.failure_bucket)}_{mode}.jsonl", rows)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
