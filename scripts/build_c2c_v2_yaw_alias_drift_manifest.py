#!/usr/bin/env python3
"""Build an acceptance manifest that separates stable alias from frame drift.

Stable alias slices become calibration-positive examples.
Jump-heavy slices become frame-drift hard cases.

This keeps the semantic ground truth clean for downstream estimator training
and for follow-up acceptance on the frame/yaw stack.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.classify_c2c_v2_yaw_alias_vs_drift import build_alias_drift_manifest


def _load_report(path: Path) -> dict:
    rep = json.loads(path.read_text(encoding="utf-8"))
    rep["report_path"] = str(path.resolve())
    return rep


def main() -> None:
    ap = argparse.ArgumentParser(description="Build a yaw alias / frame drift acceptance manifest.")
    ap.add_argument(
        "--reports",
        type=Path,
        nargs="+",
        required=True,
        help="One or more yaw_frame_sequence_report.json files.",
    )
    ap.add_argument(
        "--output_jsonl",
        type=Path,
        default=Path("runtime_artifacts/coarse2contact_v2/datasets/yaw_alias_drift_acceptance_manifest.jsonl"),
    )
    ap.add_argument(
        "--summary_json",
        type=Path,
        default=Path("runtime_artifacts/coarse2contact_v2/datasets/yaw_alias_drift_acceptance_manifest.summary.json"),
    )
    args = ap.parse_args()

    reports = [_load_report(path) for path in args.reports]
    rows, summary = build_alias_drift_manifest(reports)
    summary["report_paths"] = [str(path.resolve()) for path in args.reports]
    summary["output_jsonl"] = str(args.output_jsonl.resolve())

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_jsonl, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(args.output_jsonl)
    print(args.summary_json)


if __name__ == "__main__":
    main()
