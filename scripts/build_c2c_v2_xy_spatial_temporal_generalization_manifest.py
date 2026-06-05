#!/usr/bin/env python3
"""Build deterministic random-gate and holdout manifests for v42 XY generalization."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from prismatic.robot.coarse2contact_v2.xy_spatial_temporal_generalization import build_generalization_manifest


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _load_records(paths: list[Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in paths:
        if path.is_dir():
            for candidate in sorted(path.glob("*.jsonl")):
                records.extend(_read_jsonl(candidate))
        else:
            records.extend(_read_jsonl(path))
    records.sort(key=lambda r: (str(r.get("source_eval_root", "")), int(r.get("episode_idx", -1)), int(r.get("step_idx", r.get("step", -1)))))
    return records


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_jsonl", type=Path, nargs="+", required=True)
    ap.add_argument(
        "--output_json",
        type=Path,
        default=Path("runtime_artifacts/coarse2contact_v2/reports/c2c_v2_xy_spatial_temporal_generalization_manifest.json"),
    )
    ap.add_argument("--output_md", type=Path, default=Path("runtime_artifacts/coarse2contact_v2/reports/c2c_v2_xy_spatial_temporal_generalization_manifest.md"))
    ap.add_argument("--random_gate_size", type=int, default=10)
    ap.add_argument("--random_gate_seed", type=int, default=7)
    args = ap.parse_args()

    records = _load_records([Path(p) for p in args.dataset_jsonl])
    manifest = build_generalization_manifest(
        records,
        random_gate_size=int(args.random_gate_size),
        random_gate_seed=int(args.random_gate_seed),
    )
    manifest["source_dataset_jsonl"] = [str(p) for p in args.dataset_jsonl]

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    summary = manifest["summary"]
    lines = [
        "# C2C v2 XY Generalization Manifest",
        "",
        f"- dataset_jsonl: `{', '.join(str(p) for p in args.dataset_jsonl)}`",
        f"- random_gate_seed: `{manifest['random_gate_seed']}`",
        f"- random_gate_size: `{manifest['random_gate_size']}`",
        f"- eligible_episodes: `{summary['eligible_episodes']}`",
        f"- random10_generalization_episodes: `{summary['random10_generalization_episodes']}`",
        f"- random_holdout_pool_episodes: `{summary['random_holdout_pool_episodes']}`",
        f"- sentinel_episodes: `{json.dumps(summary['sentinel_episodes'], sort_keys=True)}`",
        "",
        "## Random10",
    ]
    for item in manifest["random10_generalization"]:
        lines.append(
            f"- ep`{int(item['episode_idx']):03d}` root=`{item['source_eval_root']}` "
            f"bucket=`{item['bucket']}` obs=`{item['observability_bucket']}` rows=`{item['rows']}`"
        )
    lines.extend(["", "## Holdout Pool"])
    for item in manifest["random_holdout_pool"][: min(20, len(manifest["random_holdout_pool"]))]:
        lines.append(
            f"- ep`{int(item['episode_idx']):03d}` root=`{item['source_eval_root']}` "
            f"bucket=`{item['bucket']}` obs=`{item['observability_bucket']}` rows=`{item['rows']}`"
        )
    if len(manifest["random_holdout_pool"]) > 20:
        lines.append(f"- ... and `{len(manifest['random_holdout_pool']) - 20}` more")
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
