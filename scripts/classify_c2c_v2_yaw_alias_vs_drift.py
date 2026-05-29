#!/usr/bin/env python3
"""Classify focused yaw/frame slices as stable alias or frame drift.

This is a lightweight offline report that turns the visual diagnosis into a
small, reusable summary.  It does not affect runtime control.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load_report(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _classify(report: dict) -> dict:
    jump_points = int(report.get("num_jump_points", 0))
    bias_corr_mae = float(report.get("bias_corrected_mae", report.get("symmetry_aware_bias_corrected_mae", 0.0)))
    symm_mae = float(report.get("symmetry_aware_mae", 0.0))
    raw_mae = float(report.get("raw_proxy_mae", 0.0))

    if jump_points == 0 and bias_corr_mae <= 0.01:
        label = "stable_alias"
        reason = "single symmetry alias explains the slice after a tiny bias correction"
    elif (jump_points > 0 and bias_corr_mae >= 0.10) or jump_points >= 2:
        label = "frame_drift"
        reason = "alias correction is not enough; the proxy flips branches across the slice"
    else:
        label = "mixed_or_unclear"
        reason = "slice needs more evidence before a hard split"

    return {
        "label": label,
        "reason": reason,
        "raw_mae": raw_mae,
        "symmetry_aware_mae": symm_mae,
        "bias_corrected_mae": bias_corr_mae,
        "jump_points": jump_points,
    }


def build_alias_drift_manifest(reports: list[dict]) -> tuple[list[dict], dict]:
    rows: list[dict] = []
    role_counts = {"calibration_positive": 0, "frame_drift_hard_case": 0, "mixed_or_unclear": 0}
    label_counts = {"stable_alias": 0, "frame_drift": 0, "mixed_or_unclear": 0}
    for rep in reports:
        cls = _classify(rep)
        if cls["label"] == "stable_alias":
            role = "calibration_positive"
        elif cls["label"] == "frame_drift":
            role = "frame_drift_hard_case"
        else:
            role = "mixed_or_unclear"
        role_counts[role] = role_counts.get(role, 0) + 1
        label_counts[cls["label"]] = label_counts.get(cls["label"], 0) + 1
        rows.append(
            {
                "schema_version": "yaw_alias_drift_acceptance_manifest_v1",
                "episode_idx": int(rep.get("episode_idx", -1)),
                "failure_bucket": str(rep.get("failure_bucket", "")),
                "primary_blocker": str(rep.get("primary_blocker", "")),
                "rows": int(rep.get("num_rows", 0)),
                "acceptance_role": role,
                "alias_label": cls["label"],
                "acceptance_reason": cls["reason"],
                "raw_mae": float(cls["raw_mae"]),
                "symmetry_aware_mae": float(cls["symmetry_aware_mae"]),
                "bias_corrected_mae": float(cls["bias_corrected_mae"]),
                "jump_points": int(cls["jump_points"]),
                "gif_path": rep.get("gif_path"),
                "jump_sheet_path": rep.get("jump_sheet_path"),
                "report_path": rep.get("report_path"),
            }
        )
    rows.sort(key=lambda r: (r["acceptance_role"], r["episode_idx"]))
    summary = {
        "schema_version": "yaw_alias_drift_acceptance_manifest_v1",
        "num_rows": int(len(rows)),
        "by_acceptance_role": role_counts,
        "by_alias_label": label_counts,
    }
    return rows, summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Classify yaw/frame slices as stable alias vs frame drift.")
    ap.add_argument(
        "--reports",
        type=Path,
        nargs="+",
        required=True,
        help="One or more yaw_frame_sequence_report.json files.",
    )
    ap.add_argument(
        "--output_dir",
        type=Path,
        default=Path("runtime_artifacts/coarse2contact_v2/reports/yaw_frame_alignment_diagnostic/alias_vs_drift"),
    )
    args = ap.parse_args()

    rows = []
    for rp in args.reports:
        rep = _load_report(rp)
        rep["report_path"] = str(rp.resolve())
        cls = _classify(rep)
        rows.append(
            {
                "episode_idx": int(rep.get("episode_idx", -1)),
                "failure_bucket": str(rep.get("failure_bucket", "")),
                "primary_blocker": str(rep.get("primary_blocker", "")),
                "rows": int(rep.get("num_rows", 0)),
                **cls,
                "gif_path": rep.get("gif_path"),
                "jump_sheet_path": rep.get("jump_sheet_path"),
                "report_path": str(rp.resolve()),
            }
        )

    rows.sort(key=lambda r: (r["label"], r["episode_idx"]))

    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "alias_vs_drift_summary.json"
    md_path = out_dir / "alias_vs_drift_summary.md"
    json_path.write_text(json.dumps({"rows": rows}, indent=2, sort_keys=True), encoding="utf-8")

    md = [
        "# Stable Alias vs Frame Drift",
        "",
        "| episode | bucket | blocker | label | raw MAE | symm MAE | bias-corr MAE | jumps |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        md.append(
            f"| ep{row['episode_idx']:03d} | {row['failure_bucket']} | {row['primary_blocker']} | {row['label']} | "
            f"{row['raw_mae']:.3f} | {row['symmetry_aware_mae']:.3f} | {row['bias_corrected_mae']:.3f} | {row['jump_points']} |"
        )
    md.extend(
        [
            "",
            "## Interpretation",
            "- stable_alias: a single symmetry-aware alias plus tiny bias correction explains the slice.",
            "- frame_drift: the proxy still flips branches even after symmetry-aware correction, so the frame definition is not stable enough.",
            "- mixed_or_unclear: useful for follow-up, but not yet a hard split.",
        ]
    )
    md_path.write_text("\n".join(md), encoding="utf-8")
    print(json.dumps({"json_path": str(json_path), "md_path": str(md_path), "rows": rows}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
