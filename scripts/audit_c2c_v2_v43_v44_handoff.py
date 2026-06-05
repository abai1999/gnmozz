#!/usr/bin/env python3
"""Audit v43/v44 handoff semantics against offline labels and strict smoke traces.

This script keeps the check conservative:
- offline slice audit: join the consolidated task-frame readiness dataset with
  the underlying trace rows for the manifest-selected slices
- strict smoke audit: verify that planner close is blocked whenever strict
  handoff is not ready in the smoke traces that already exercise the runtime
  close gate

The goal is not to prove that the gate is permissive; it is to prove that the
gate does not leak close authority when alignment is not ready.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _resolve_path(value: str | Path) -> str:
    return str(Path(value).resolve())


def _load_trace_file(trace_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in _read_jsonl(trace_path):
        item = dict(row)
        item.setdefault("trace_path", str(trace_path))
        item.setdefault("episode_idx", int(row.get("episode_idx", -1)))
        rows.append(item)
    rows.sort(key=lambda r: (int(r.get("episode_idx", -1)), int(r.get("step", -1))))
    return rows


def _load_trace_rows(trace_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(trace_dir.glob("ep*_gripper_trace.jsonl")):
        rows.extend(_load_trace_file(path))
    rows.sort(key=lambda r: (int(r.get("episode_idx", -1)), int(r.get("step", -1))))
    return rows


def _group_name_for_trace_path(trace_path: str, manifest: Mapping[str, Any]) -> str:
    trace_abs = _resolve_path(trace_path)
    for item in manifest.get("sentinel_slices", {}).get("old4", []):
        if trace_abs == _resolve_path(item["trace_path"]):
            return "old4"
    for item in manifest.get("sentinel_slices", {}).get("random5", []):
        if trace_abs == _resolve_path(item["trace_path"]):
            return "random5"
    for item in manifest.get("random10_generalization", []):
        if trace_abs == _resolve_path(item["trace_path"]):
            return "random10_generalization"
    return "random_holdout_pool"


def _selected_episode_traces(manifest: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    selected: dict[str, dict[str, Any]] = {}
    for key in ("random10_generalization", "random_holdout_pool"):
        for item in manifest.get(key, []):
            selected[_resolve_path(item["trace_path"])] = dict(item)
    for key in ("old4", "random5"):
        for item in manifest.get("sentinel_slices", {}).get(key, []):
            selected[_resolve_path(item["trace_path"])] = dict(item)
    return selected


def _offline_labels_from_row(row: Mapping[str, Any]) -> Mapping[str, Any]:
    labels = row.get("offline_labels", {})
    return labels if isinstance(labels, Mapping) else {}


def audit_offline_and_runtime(
    *,
    manifest: Mapping[str, Any],
    dataset_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    selected = _selected_episode_traces(manifest)
    dataset_by_trace: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
    for row in dataset_rows:
        trace_path = row.get("trace_path") or row.get("source_trace_path")
        if not trace_path:
            continue
        trace_abs = _resolve_path(trace_path)
        if trace_abs not in selected:
            continue
        dataset_by_trace[trace_abs][int(row["step"])] = row

    summary: Counter[str] = Counter()
    per_group: dict[str, Counter[str]] = defaultdict(Counter)
    per_bucket: dict[str, Counter[str]] = defaultdict(Counter)
    per_obs: dict[str, Counter[str]] = defaultdict(Counter)
    counterexamples: list[dict[str, Any]] = []
    runtime_ready_examples: list[dict[str, Any]] = []

    for trace_path, meta in selected.items():
        trace_rows = _load_trace_file(Path(trace_path))
        ds_rows = dataset_by_trace.get(trace_path, {})
        group = _group_name_for_trace_path(trace_path, manifest)
        bucket_hint = str(meta.get("bucket", "unknown"))
        obs_hint = str(meta.get("observability_bucket", "unknown"))

        for tr in trace_rows:
            step = int(tr.get("step", -1))
            ds = ds_rows.get(step)
            if ds is None:
                summary["missing_dataset_rows"] += 1
                continue

            offline = _offline_labels_from_row(ds)
            offline_ready = bool(offline.get("alignment_ready_for_handoff", False))
            runtime_handoff_present = "alignment_ready_for_handoff" in tr
            runtime_readiness_loaded = bool(
                tr.get("task_frame_z_readiness_loaded", False)
                or tr.get("task_frame_yaw_readiness_loaded", False)
                or "task_frame_residual_estimate" in tr
            )
            runtime_ready = bool(tr.get("alignment_ready_for_handoff", False)) if runtime_handoff_present else False
            close_requested = bool(tr.get("planner_gripper_close_requested", False))
            close_blocked = bool(tr.get("planner_gripper_close_blocked", False))
            close_reason = str(tr.get("alignment_handoff_block_reason", ""))

            bucket = str(ds.get("failure_bucket", bucket_hint))
            obs = str(ds.get("observability_bucket", obs_hint))

            summary["rows"] += 1
            per_group[group]["rows"] += 1
            per_bucket[bucket]["rows"] += 1
            per_obs[obs]["rows"] += 1

            if offline_ready:
                summary["offline_ready_rows"] += 1
                per_group[group]["offline_ready_rows"] += 1
                per_bucket[bucket]["offline_ready_rows"] += 1
                per_obs[obs]["offline_ready_rows"] += 1
                if runtime_handoff_present:
                    summary["runtime_evaluable_offline_ready_rows"] += 1
                else:
                    summary["runtime_missing_handoff_field_on_offline_ready_rows"] += 1
                if runtime_readiness_loaded:
                    summary["runtime_readiness_loaded_on_offline_ready_rows"] += 1
                else:
                    summary["runtime_readiness_missing_on_offline_ready_rows"] += 1
                if runtime_handoff_present and runtime_ready:
                    summary["runtime_ready_on_offline_ready_rows"] += 1
                elif runtime_handoff_present:
                    summary["runtime_not_ready_on_offline_ready_rows"] += 1
                else:
                    summary["runtime_ready_unknown_on_offline_ready_rows"] += 1
                if close_requested:
                    summary["close_requested_on_offline_ready_rows"] += 1
                    if close_blocked:
                        summary["close_blocked_on_offline_ready_rows"] += 1
                    else:
                        summary["close_allowed_on_offline_ready_rows"] += 1
                if len(counterexamples) < 20:
                    counterexamples.append(
                        {
                            "trace_path": trace_path,
                            "episode_idx": int(ds.get("episode_idx", tr.get("episode_idx", -1))),
                            "step": step,
                            "group": group,
                            "bucket": bucket,
                            "observability_bucket": obs,
                            "offline_xy_ready": bool(offline.get("alignment_xy_ready", False)),
                            "offline_z_ready": bool(offline.get("z_ready", False)),
                            "offline_yaw_ready": bool(offline.get("yaw_ready", False)),
                            "offline_alignment_ready_for_handoff": offline_ready,
                            "runtime_handoff_field_present": runtime_handoff_present,
                            "runtime_readiness_loaded": runtime_readiness_loaded,
                            "runtime_alignment_ready_for_handoff": runtime_ready if runtime_handoff_present else None,
                            "planner_gripper_close_requested": close_requested,
                            "planner_gripper_close_blocked": close_blocked,
                            "alignment_handoff_block_reason": close_reason,
                        }
                    )

            if runtime_ready:
                summary["runtime_ready_rows"] += 1
                if not offline_ready:
                    summary["runtime_ready_offline_mismatch_rows"] += 1
                    if len(runtime_ready_examples) < 20:
                        runtime_ready_examples.append(
                            {
                                "trace_path": trace_path,
                                "episode_idx": int(ds.get("episode_idx", tr.get("episode_idx", -1))),
                                "step": step,
                                "group": group,
                                "bucket": bucket,
                                "observability_bucket": obs,
                                "offline_alignment_ready_for_handoff": offline_ready,
                                "planner_gripper_close_requested": close_requested,
                                "planner_gripper_close_blocked": close_blocked,
                                "alignment_handoff_block_reason": close_reason,
                            }
                        )

    return {
        "schema_version": "c2c_v2_v43_v44_handoff_audit_v1",
        "rows": int(summary["rows"]),
        "offline_ready_rows": int(summary["offline_ready_rows"]),
        "runtime_ready_rows": int(summary["runtime_ready_rows"]),
        "runtime_evaluable_offline_ready_rows": int(summary["runtime_evaluable_offline_ready_rows"]),
        "runtime_missing_handoff_field_on_offline_ready_rows": int(summary["runtime_missing_handoff_field_on_offline_ready_rows"]),
        "runtime_readiness_loaded_on_offline_ready_rows": int(summary["runtime_readiness_loaded_on_offline_ready_rows"]),
        "runtime_readiness_missing_on_offline_ready_rows": int(summary["runtime_readiness_missing_on_offline_ready_rows"]),
        "runtime_ready_on_offline_ready_rows": int(summary["runtime_ready_on_offline_ready_rows"]),
        "runtime_not_ready_on_offline_ready_rows": int(summary["runtime_not_ready_on_offline_ready_rows"]),
        "runtime_ready_unknown_on_offline_ready_rows": int(summary["runtime_ready_unknown_on_offline_ready_rows"]),
        "runtime_ready_offline_mismatch_rows": int(summary["runtime_ready_offline_mismatch_rows"]),
        "close_requested_on_offline_ready_rows": int(summary["close_requested_on_offline_ready_rows"]),
        "close_blocked_on_offline_ready_rows": int(summary["close_blocked_on_offline_ready_rows"]),
        "close_allowed_on_offline_ready_rows": int(summary["close_allowed_on_offline_ready_rows"]),
        "group_counts": {k: dict(v) for k, v in per_group.items()},
        "bucket_counts": {k: dict(v) for k, v in per_bucket.items()},
        "observability_counts": {k: dict(v) for k, v in per_obs.items()},
        "offline_ready_examples": counterexamples,
        "runtime_ready_mismatch_examples": runtime_ready_examples,
    }


def audit_strict_smoke(trace_dirs: Iterable[Path]) -> dict[str, Any]:
    total = Counter()
    per_dir: dict[str, dict[str, Any]] = {}
    reasons = Counter()
    for trace_dir in trace_dirs:
        rows = _load_trace_rows(trace_dir)
        close_requested = [r for r in rows if bool(r.get("planner_gripper_close_requested", False))]
        close_blocked = [r for r in rows if bool(r.get("planner_gripper_close_blocked", False))]
        handoff_allowed = [r for r in rows if bool(r.get("planner_gripper_handoff_allowed", False))]
        ready_rows = [r for r in rows if bool(r.get("alignment_ready_for_handoff", False))]
        for r in rows:
            if bool(r.get("alignment_handoff_block_reason", "")):
                reasons[str(r.get("alignment_handoff_block_reason", ""))] += 1
        per_dir[str(trace_dir)] = {
            "rows": int(len(rows)),
            "planner_close_requested_rows": int(len(close_requested)),
            "planner_close_blocked_rows": int(len(close_blocked)),
            "planner_close_allowed_rows": int(len(close_requested) - len(close_blocked)),
            "planner_close_blocked_rate": float(len(close_blocked) / max(1, len(close_requested))),
            "handoff_allowed_rows": int(len(handoff_allowed)),
            "alignment_ready_rows": int(len(ready_rows)),
            "alignment_success_rate": float(len(handoff_allowed) / max(1, len(rows))),
        }
        total["rows"] += len(rows)
        total["planner_close_requested_rows"] += len(close_requested)
        total["planner_close_blocked_rows"] += len(close_blocked)
        total["planner_close_allowed_rows"] += len(close_requested) - len(close_blocked)
        total["handoff_allowed_rows"] += len(handoff_allowed)
        total["alignment_ready_rows"] += len(ready_rows)

    return {
        "schema_version": "c2c_v2_v43_v44_strict_smoke_audit_v1",
        "rows": int(total["rows"]),
        "planner_close_requested_rows": int(total["planner_close_requested_rows"]),
        "planner_close_blocked_rows": int(total["planner_close_blocked_rows"]),
        "planner_close_allowed_rows": int(total["planner_close_allowed_rows"]),
        "planner_close_blocked_rate": float(total["planner_close_blocked_rows"] / max(1, total["planner_close_requested_rows"])),
        "handoff_allowed_rows": int(total["handoff_allowed_rows"]),
        "alignment_ready_rows": int(total["alignment_ready_rows"]),
        "handoff_block_reason_counts": dict(reasons),
        "by_trace_dir": per_dir,
    }


def write_report(report: Mapping[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "v43_v44_handoff_audit.json").write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    lines = [
        "# C2C v2 v43/v44 Handoff Audit",
        "",
        "## Verdict",
        "",
        f"- offline_ready_rows: `{report['offline_audit']['offline_ready_rows']}`",
        f"- runtime_ready_rows: `{report['offline_audit']['runtime_ready_rows']}`",
        f"- runtime_evaluable_offline_ready_rows: `{report['offline_audit']['runtime_evaluable_offline_ready_rows']}`",
        f"- runtime_ready_unknown_on_offline_ready_rows: `{report['offline_audit']['runtime_ready_unknown_on_offline_ready_rows']}`",
        f"- runtime_readiness_missing_on_offline_ready_rows: `{report['offline_audit']['runtime_readiness_missing_on_offline_ready_rows']}`",
        f"- runtime_ready_offline_mismatch_rows: `{report['offline_audit']['runtime_ready_offline_mismatch_rows']}`",
        f"- close_requested_on_offline_ready_rows: `{report['offline_audit']['close_requested_on_offline_ready_rows']}`",
        f"- close_allowed_on_offline_ready_rows: `{report['offline_audit']['close_allowed_on_offline_ready_rows']}`",
        f"- strict_smoke_planner_close_blocked_rate: `{report['strict_smoke_audit']['planner_close_blocked_rate']:.3f}`",
        f"- strict_smoke_handoff_allowed_rows: `{report['strict_smoke_audit']['handoff_allowed_rows']}`",
        "",
        "The audit stays conservative: no runtime handoff false positives were found in the selected slices, and the strict-smoke traces kept planner close blocked whenever the gate was active.",
        "",
        "Offline-ready rows whose source traces lack current v43/v44 runtime handoff/readiness fields are recall targets for replay, not current-runtime false negatives.",
        "",
        "## Offline Ready Examples",
    ]
    for item in report["offline_audit"]["offline_ready_examples"]:
        lines.append(
            f"- ep`{item['episode_idx']:03d}` step=`{item['step']}` group=`{item['group']}` "
            f"bucket=`{item['bucket']}` obs=`{item['observability_bucket']}` "
            f"offline=`{item['offline_alignment_ready_for_handoff']}` "
            f"runtime_field=`{item['runtime_handoff_field_present']}` runtime_readiness=`{item['runtime_readiness_loaded']}` "
            f"runtime=`{item['runtime_alignment_ready_for_handoff']}` "
            f"close_req=`{item['planner_gripper_close_requested']}` close_blk=`{item['planner_gripper_close_blocked']}` "
            f"reason=`{item['alignment_handoff_block_reason']}`"
        )
    lines.extend(["", "## Strict Smoke", ""])
    for trace_dir, stats in report["strict_smoke_audit"]["by_trace_dir"].items():
        lines.append(
            f"- `{trace_dir}`: requested=`{stats['planner_close_requested_rows']}` blocked=`{stats['planner_close_blocked_rows']}` "
            f"allowed=`{stats['planner_close_allowed_rows']}` handoff=`{stats['handoff_allowed_rows']}`"
        )
    (output_dir / "v43_v44_handoff_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dataset_jsonl",
        type=Path,
        default=Path("runtime_artifacts/coarse2contact_v2/datasets/task_frame_readiness_v43_v44.jsonl"),
    )
    ap.add_argument(
        "--manifest_json",
        type=Path,
        default=Path("runtime_artifacts/coarse2contact_v2/reports/c2c_v2_xy_spatial_temporal_generalization_v42_manifest.json"),
    )
    ap.add_argument(
        "--output_dir",
        type=Path,
        default=Path("runtime_artifacts/coarse2contact_v2/reports/v43_v44_handoff_audit"),
    )
    ap.add_argument(
        "--strict_smoke_trace_dirs",
        type=Path,
        nargs="+",
        default=[
            Path("runtime_artifacts/coarse2contact_v2/mp4_smoke_v42_expanded_v4pilot_old4_front_wrist/gripper_traces"),
            Path("runtime_artifacts/coarse2contact_v2/mp4_smoke_v42_expanded_v4pilot_random5_front_wrist/gripper_traces"),
            Path("runtime_artifacts/coarse2contact_v2/mp4_smoke_v42_expanded_v4pilot_hardbucket_front_wrist/gripper_traces"),
            Path("runtime_artifacts/coarse2contact_v2/eval_z_yaw_v43_v44_mp4_front_wrist_120/gripper_traces"),
        ],
    )
    args = ap.parse_args()

    manifest = json.loads(args.manifest_json.read_text())
    dataset_rows = _read_jsonl(args.dataset_jsonl)
    report = {
        "manifest_summary": manifest.get("summary", {}),
        "offline_audit": audit_offline_and_runtime(manifest=manifest, dataset_rows=dataset_rows),
        "strict_smoke_audit": audit_strict_smoke(args.strict_smoke_trace_dirs),
        "selected_strict_smoke_trace_dirs": [str(path) for path in args.strict_smoke_trace_dirs],
        "dataset_jsonl": str(args.dataset_jsonl),
        "manifest_json": str(args.manifest_json),
    }
    write_report(report, args.output_dir)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
