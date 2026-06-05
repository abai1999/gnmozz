"""Helpers for v42 XY spatial-temporal generalization splits and gates."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from collections import defaultdict, Counter
from typing import Any, Mapping


def _as_str(value: Any, default: str = "") -> str:
    text = str(value or "").strip()
    return text or str(default)


def source_eval_root_key(row: Mapping[str, Any]) -> str:
    """Return the stable root/session key for held-out splitting."""

    source_root = _as_str(row.get("source_eval_root", ""))
    if source_root:
        return source_root
    sequence = _as_str(row.get("sequence_id", ""))
    if sequence:
        return sequence.split("::", 1)[0]
    source_trace = _as_str(row.get("source_trace_path", row.get("trace_path", "")))
    if source_trace:
        return source_trace
    runtime_obs = _as_str(row.get("runtime_obs_path", row.get("npz_path", "")))
    if runtime_obs:
        return runtime_obs.rsplit("/", 1)[0]
    episode_idx = int(row.get("episode_idx", -1))
    return f"ep{episode_idx:03d}"


def episode_identity(row: Mapping[str, Any]) -> tuple[str, int]:
    return source_eval_root_key(row), int(row.get("episode_idx", -1))


def _hash_rank(text: str, seed: int) -> int:
    digest = hashlib.sha1(f"{seed}:{text}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def group_records_by_root(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        grouped[source_eval_root_key(row)].append(dict(row))
    for root in grouped:
        grouped[root].sort(key=lambda r: int(r.get("step_idx", r.get("step", -1))))
    return dict(sorted(grouped.items()))


def group_records_by_episode(records: list[dict[str, Any]]) -> dict[tuple[str, int], list[dict[str, Any]]]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        grouped[episode_identity(row)].append(dict(row))
    for key in grouped:
        grouped[key].sort(key=lambda r: int(r.get("step_idx", r.get("step", -1))))
    return dict(sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1])))


@dataclass(frozen=True)
class SourceRootSplit:
    split_mode: str
    train_records: list[dict[str, Any]]
    val_records: list[dict[str, Any]]
    test_records: list[dict[str, Any]]
    train_source_eval_roots: list[str]
    val_source_eval_roots: list[str]
    test_source_eval_roots: list[str]


def split_records_by_source_root(
    records: list[dict[str, Any]],
    *,
    split_mode: str = "auto",
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
    seed: int = 11,
    train_source_eval_roots: set[str] | None = None,
    val_source_eval_roots: set[str] | None = None,
    test_source_eval_roots: set[str] | None = None,
) -> SourceRootSplit:
    """Split records by stable source root/session, keeping roots intact."""

    train_source_eval_roots = {str(item).strip() for item in (train_source_eval_roots or set()) if str(item).strip()}
    val_source_eval_roots = {str(item).strip() for item in (val_source_eval_roots or set()) if str(item).strip()}
    test_source_eval_roots = {str(item).strip() for item in (test_source_eval_roots or set()) if str(item).strip()}
    root_keys = sorted({source_eval_root_key(row) for row in records})
    mode = str(split_mode or "auto").strip().lower()
    if mode not in {"auto", "root", "episode"}:
        raise ValueError(f"unknown split_mode: {split_mode}")
    if mode == "episode" or (mode == "auto" and not any(_as_str(row.get("source_eval_root", "")) for row in records)):
        return _split_by_episode(records, val_fraction=val_fraction, test_fraction=test_fraction, seed=seed)

    shuffled = list(root_keys)
    shuffled.sort(key=lambda item: _hash_rank(item, seed))

    if train_source_eval_roots:
        train_source_eval_roots = {root for root in train_source_eval_roots if root in root_keys}
        val_source_eval_roots.difference_update(train_source_eval_roots)
        test_source_eval_roots.difference_update(train_source_eval_roots)

    if not test_source_eval_roots:
        candidate_test_roots = [root for root in shuffled if root not in train_source_eval_roots]
        n_test = max(0, int(round(len(candidate_test_roots) * float(test_fraction))))
        if n_test > 0 and len(candidate_test_roots) > 1:
            test_source_eval_roots = set(candidate_test_roots[:n_test])
    if not val_source_eval_roots:
        remaining = [root for root in shuffled if root not in test_source_eval_roots and root not in train_source_eval_roots]
        n_val = max(1 if remaining else 0, int(round(len(remaining) * float(val_fraction)))) if remaining else 0
        if n_val > 0 and remaining:
            val_source_eval_roots = set(remaining[:n_val])

    train_source_eval_roots = {
        root
        for root in root_keys
        if root not in val_source_eval_roots and root not in test_source_eval_roots
    } | train_source_eval_roots
    if not train_source_eval_roots:
        # Preserve at least one training root by shrinking the smaller held-out side.
        if test_source_eval_roots:
            test_source_eval_roots = set(list(test_source_eval_roots)[1:])
        if not test_source_eval_roots and val_source_eval_roots:
            val_source_eval_roots = set(list(val_source_eval_roots)[1:])
        train_source_eval_roots = {root for root in root_keys if root not in val_source_eval_roots and root not in test_source_eval_roots}

    train_records = [r for r in records if source_eval_root_key(r) in train_source_eval_roots]
    val_records = [r for r in records if source_eval_root_key(r) in val_source_eval_roots]
    test_records = [r for r in records if source_eval_root_key(r) in test_source_eval_roots]
    return SourceRootSplit(
        split_mode="root",
        train_records=train_records,
        val_records=val_records,
        test_records=test_records,
        train_source_eval_roots=sorted(train_source_eval_roots),
        val_source_eval_roots=sorted(val_source_eval_roots),
        test_source_eval_roots=sorted(test_source_eval_roots),
    )


def _split_by_episode(
    records: list[dict[str, Any]],
    *,
    val_fraction: float,
    test_fraction: float,
    seed: int,
) -> SourceRootSplit:
    episodes = sorted({int(r.get("episode_idx", -1)) for r in records if int(r.get("episode_idx", -1)) >= 0})
    shuffled = list(episodes)
    shuffled.sort(key=lambda item: _hash_rank(f"ep{item:03d}", seed))
    n_test = max(0, int(round(len(shuffled) * float(test_fraction))))
    n_val = max(1 if len(shuffled) > n_test else 0, int(round(len(shuffled) * float(val_fraction))))
    test_eps = set(shuffled[:n_test])
    val_eps = set(shuffled[n_test : n_test + n_val])
    train_eps = {ep for ep in episodes if ep not in test_eps and ep not in val_eps}
    return SourceRootSplit(
        split_mode="episode",
        train_records=[r for r in records if int(r.get("episode_idx", -1)) in train_eps],
        val_records=[r for r in records if int(r.get("episode_idx", -1)) in val_eps],
        test_records=[r for r in records if int(r.get("episode_idx", -1)) in test_eps],
        train_source_eval_roots=sorted({source_eval_root_key(r) for r in records if int(r.get("episode_idx", -1)) in train_eps}),
        val_source_eval_roots=sorted({source_eval_root_key(r) for r in records if int(r.get("episode_idx", -1)) in val_eps}),
        test_source_eval_roots=sorted({source_eval_root_key(r) for r in records if int(r.get("episode_idx", -1)) in test_eps}),
    )


def _summarize_episode_meta(rows: list[dict[str, Any]]) -> dict[str, Any]:
    first = rows[0] if rows else {}
    bucket_counts = Counter(str(r.get("failure_bucket", r.get("bucket", "unknown")) or "unknown") for r in rows)
    obs_counts = Counter(str(r.get("observability_bucket", "unknown") or "unknown") for r in rows)
    return {
        "source_eval_root": source_eval_root_key(first) if first else "",
        "episode_idx": int(first.get("episode_idx", -1)) if first else -1,
        "sequence_id": _as_str(first.get("sequence_id", "")),
        "trace_path": _as_str(first.get("trace_path", "")),
        "runtime_obs_path": _as_str(first.get("runtime_obs_path", first.get("npz_path", ""))),
        "bucket": bucket_counts.most_common(1)[0][0] if bucket_counts else "unknown",
        "observability_bucket": obs_counts.most_common(1)[0][0] if obs_counts else "unknown",
        "rows": int(len(rows)),
        "active_rows": int(sum(1 for r in rows if bool(r.get("grasp_probe_active", False)))),
    }


def build_generalization_manifest(
    records: list[dict[str, Any]],
    *,
    random_gate_size: int = 10,
    random_gate_seed: int = 7,
    sentinel_root_substrings: tuple[str, ...] = (
        "mp4_smoke_v38_alignment_lifecycle_runtime_xy_mlp_temporal_old4",
        "mp4_smoke_v38_alignment_lifecycle_runtime_xy_mlp_temporal_random5",
    ),
) -> dict[str, Any]:
    """Build a deterministic random gate and holdout pool from episode groups."""

    episode_groups = group_records_by_episode(records)
    episode_meta: list[dict[str, Any]] = []
    sentinel_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    eligible_groups: list[dict[str, Any]] = []
    for (root, episode_idx), rows in episode_groups.items():
        meta = _summarize_episode_meta(rows)
        meta["episode_key"] = {"source_eval_root": root, "episode_idx": int(episode_idx)}
        meta["selection_reason"] = "eligible"
        if any(substr in root for substr in sentinel_root_substrings):
            meta["selection_reason"] = "sentinel_root"
            if "old4" in root:
                sentinel_groups["old4"].append(meta)
            elif "random5" in root:
                sentinel_groups["random5"].append(meta)
            else:
                sentinel_groups["sentinel"].append(meta)
        else:
            eligible_groups.append(meta)
        episode_meta.append(meta)

    roots = sorted({str(item["source_eval_root"]) for item in eligible_groups})
    roots.sort(key=lambda item: _hash_rank(item, random_gate_seed))
    by_root: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in eligible_groups:
        by_root[str(item["source_eval_root"])].append(item)
    for root in by_root:
        by_root[root].sort(key=lambda item: _hash_rank(f"{item['source_eval_root']}::ep{int(item['episode_idx']):03d}", random_gate_seed))

    gate: list[dict[str, Any]] = []
    pool: list[dict[str, Any]] = []
    root_positions = {root: 0 for root in roots}
    while len(gate) < int(max(0, random_gate_size)) and roots:
        progressed = False
        for root in roots:
            idx = root_positions[root]
            if idx >= len(by_root[root]):
                continue
            gate.append(dict(by_root[root][idx]))
            root_positions[root] = idx + 1
            progressed = True
            if len(gate) >= int(max(0, random_gate_size)):
                break
        if not progressed:
            break
    chosen_keys = {
        (str(item["source_eval_root"]), int(item["episode_idx"]))
        for item in gate
    }
    for item in eligible_groups:
        key = (str(item["source_eval_root"]), int(item["episode_idx"]))
        if key not in chosen_keys:
            pool.append(dict(item))

    return {
        "schema_version": "c2c_v2_xy_spatial_temporal_generalization_manifest_v1",
        "random_gate_seed": int(random_gate_seed),
        "random_gate_size": int(random_gate_size),
        "random10_generalization": gate,
        "random_holdout_pool": pool,
        "sentinel_slices": {k: v for k, v in sorted(sentinel_groups.items())},
        "groups": {
            "random10_generalization": gate,
            "random_holdout_pool": pool,
            **{f"sentinel_{k}": v for k, v in sorted(sentinel_groups.items())},
        },
        "all_episodes": episode_meta,
        "summary": {
            "all_episodes": int(len(episode_meta)),
            "eligible_episodes": int(len(eligible_groups)),
            "random10_generalization_episodes": int(len(gate)),
            "random_holdout_pool_episodes": int(len(pool)),
            "sentinel_episodes": {k: int(len(v)) for k, v in sorted(sentinel_groups.items())},
            "eligible_roots": roots,
            "group_names": [
                "random10_generalization",
                "random_holdout_pool",
                *[f"sentinel_{k}" for k in sorted(sentinel_groups)],
            ],
        },
    }
