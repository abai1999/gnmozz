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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--dataset_report", default=None)
    ap.add_argument("--teacher_audit", default=None)
    ap.add_argument("--stagea_gate", default=None)
    ap.add_argument("--stageb_gate", default=None)
    ap.add_argument("--candidate_compare", default=None)
    args = ap.parse_args()

    summary = {
        "root": str(Path(args.root).resolve()),
        "dataset_report": _load_json(args.dataset_report),
        "teacher_audit": _load_json(args.teacher_audit),
        "stagea_gate": _load_json(args.stagea_gate),
        "stageb_gate": _load_json(args.stageb_gate),
        "candidate_compare": _load_json(args.candidate_compare),
    }
    if summary["stagea_gate"]:
        summary["decision"] = summary["stagea_gate"].get("decision", "unknown")
    elif summary["stageb_gate"]:
        summary["decision"] = summary["stageb_gate"].get("decision", "unknown")
    else:
        summary["decision"] = "unknown"

    if summary["candidate_compare"]:
        summary["main_candidate"] = summary["candidate_compare"].get("main_candidate", {})
        summary["comparison_baseline"] = summary["candidate_compare"].get("comparison_baseline", {})
        summary["candidate_compare_decision"] = summary["candidate_compare"].get("decision", "unknown")
    else:
        summary["main_candidate"] = {
            "role": "stageA",
            "source": "stageA best deploy candidate",
            "gate_report": args.stagea_gate,
        }
        summary["comparison_baseline"] = {
            "role": "stageB",
            "source": "progress-heavy comparison baseline",
            "gate_report": args.stageb_gate,
        }

    out = Path(args.root) / "alignment_v3_summary.json"
    out.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
