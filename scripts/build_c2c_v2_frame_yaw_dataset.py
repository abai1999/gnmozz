#!/usr/bin/env python3
"""Build a frame-yaw estimator dataset from C2C v2 frame residual relabels."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from prismatic.robot.coarse2contact_v2.frame_yaw_estimator import (
    FRAME_YAW_FEATURE_NAMES,
    frame_yaw_feature_vector,
    frame_yaw_label_from_row,
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _expand_inputs(paths: list[Path]) -> list[Path]:
    out: list[Path] = []
    for path in paths:
        if path.is_dir():
            out.extend(sorted(path.rglob("frame_residual_v2.jsonl")))
        elif path.is_file():
            out.append(path)
    dedup: list[Path] = []
    seen: set[str] = set()
    for path in out:
        key = str(path.resolve())
        if key not in seen:
            seen.add(key)
            dedup.append(path)
    return dedup


def _row_mapping(row: dict[str, Any], key: str) -> dict[str, Any]:
    value = row.get(key)
    return dict(value) if isinstance(value, dict) else {}


def _row_bool(row: dict[str, Any], key: str, default: bool = False) -> bool:
    value = row.get(key, default)
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, (int, float, np.integer, np.floating)):
        return bool(float(value) > 0.5)
    return bool(value)


def _visual_class(row: dict[str, Any]) -> str:
    obs = _row_mapping(row, "obs_t")
    return str(obs.get("visual_observability_class", row.get("visual_observability_class", "")))


def _yaw_observable(row: dict[str, Any]) -> bool:
    return _row_bool(row, "yaw_control_observable", _row_bool(row, "yaw_observable", False))


def _hard_yaw_negative(row: dict[str, Any]) -> bool:
    visual = _visual_class(row) == "visual_observable"
    near = _row_bool(row, "near_basin_shell", False)
    entry = _row_bool(row, "yaw_entry_feasible", False)
    return bool(visual and (near or entry) and not _yaw_observable(row))


def _yaw_stratum(row: dict[str, Any], yaw_observable: float) -> str:
    visual = _visual_class(row) == "visual_observable"
    entry = _row_bool(row, "yaw_entry_feasible", False)
    near = _row_bool(row, "near_basin_shell", False)
    if float(yaw_observable) > 0.5:
        if visual and entry and near:
            return "pos_visual_entry_near"
        if visual and entry:
            return "pos_visual_entry"
        if visual:
            return "pos_visual_other"
        return "pos_other"
    if visual and entry and near:
        return "neg_visual_entry_near"
    if visual and entry:
        return "neg_visual_entry"
    if visual:
        return "neg_visual_other"
    return "neg_other"


def _balanced_sample_weights(yaw: np.ndarray, stratum: np.ndarray, focus: np.ndarray) -> np.ndarray:
    n = int(yaw.shape[0])
    if n <= 0:
        return np.zeros((0,), dtype=np.float32)
    weights = np.ones((n,), dtype=np.float32)
    pos = yaw > 0.5
    pos_count = max(int(np.count_nonzero(pos)), 1)
    neg_count = max(n - int(np.count_nonzero(pos)), 1)
    weights[pos] *= min(float(n) / (2.0 * float(pos_count)), 30.0)
    weights[~pos] *= min(float(n) / (2.0 * float(neg_count)), 5.0)
    for name in np.unique(stratum):
        mask = stratum == name
        if not np.any(mask):
            continue
        weights[mask] *= min((float(n) / float(np.count_nonzero(mask))) ** 0.25, 4.0)
    weights[focus > 0.5] *= 2.0
    return weights.astype(np.float32)


def _balance_observability_pool(
    indexed_rows: list[tuple[int, dict[str, Any], float, float, float]],
    *,
    ratio: float,
    seed: int,
) -> list[tuple[int, dict[str, Any], float, float, float]]:
    positives = [item for item in indexed_rows if _yaw_observable(item[1])]
    hard_negatives = [item for item in indexed_rows if (not _yaw_observable(item[1])) and _hard_yaw_negative(item[1])]
    easy_negatives = [item for item in indexed_rows if (not _yaw_observable(item[1])) and (not _hard_yaw_negative(item[1]))]
    if not positives:
        return indexed_rows
    rng = np.random.default_rng(int(seed))
    desired_neg = int(round(len(positives) * float(ratio)))
    selected_neg: list[tuple[int, dict[str, Any], float, float, float]] = []
    if hard_negatives:
        order = np.arange(len(hard_negatives), dtype=np.int64)
        rng.shuffle(order)
        take_hard = min(len(hard_negatives), desired_neg)
        selected_neg.extend([hard_negatives[int(i)] for i in order[:take_hard]])
    if len(selected_neg) < desired_neg and easy_negatives:
        need = desired_neg - len(selected_neg)
        order = np.arange(len(easy_negatives), dtype=np.int64)
        rng.shuffle(order)
        selected_neg.extend([easy_negatives[int(i)] for i in order[:need]])
    selected = positives + selected_neg
    selected.sort(key=lambda item: item[0])
    return selected


def _assign_balanced_split(
    *,
    episode_idx: np.ndarray,
    yaw_observable: np.ndarray,
    yaw_positive_focus: np.ndarray,
    stratum: np.ndarray,
    val_ratio: float,
    seed: int,
    min_val_yaw_positive: int,
    min_val_focus_positive: int,
) -> np.ndarray:
    n = int(episode_idx.shape[0])
    split = np.full((n,), "train", dtype="<U8")
    if n <= 0:
        return split
    rng = np.random.default_rng(int(seed))
    val_eps: set[int] = set()
    all_eps = np.unique(episode_idx.astype(np.int64))

    for name in np.unique(stratum):
        eps = np.unique(episode_idx[stratum == name].astype(np.int64))
        if eps.size <= 0:
            continue
        rng.shuffle(eps)
        take = max(1, int(round(float(val_ratio) * eps.size)))
        if eps.size > 1:
            take = min(take, eps.size - 1)
        val_eps.update(int(x) for x in eps[:take])

    pos_eps = np.unique(episode_idx[yaw_observable > 0.5].astype(np.int64))
    focus_eps = np.unique(episode_idx[yaw_positive_focus > 0.5].astype(np.int64))
    for eps, wanted_rows, mask in (
        (focus_eps, int(min_val_focus_positive), yaw_positive_focus > 0.5),
        (pos_eps, int(min_val_yaw_positive), yaw_observable > 0.5),
    ):
        if eps.size <= 0:
            continue
        shuffled = np.array(eps, dtype=np.int64)
        rng.shuffle(shuffled)
        for ep in shuffled:
            if int(np.count_nonzero(mask & np.isin(episode_idx, list(val_eps)))) >= wanted_rows:
                break
            if eps.size > 1 and int(ep) not in val_eps:
                val_eps.add(int(ep))

    if len(val_eps) >= all_eps.size and all_eps.size > 1:
        # Keep at least one episode with yaw-positive rows in train when possible.
        train_candidate_eps = [int(ep) for ep in pos_eps if int(ep) in val_eps]
        if train_candidate_eps:
            val_eps.remove(train_candidate_eps[-1])
        else:
            val_eps.remove(int(all_eps[-1]))

    val_mask = np.isin(episode_idx, list(val_eps))
    split[val_mask] = "val"
    return split


def build_dataset(
    rows: list[dict[str, Any]],
    *,
    stage_name: str = "",
    skill_type: str = "",
    require_finite_label: bool = True,
    balanced_split: bool = False,
    balance_observability_pool: bool = False,
    observability_negative_to_positive_ratio: float = 3.0,
    val_ratio: float = 0.2,
    seed: int = 7,
    min_val_yaw_positive: int = 30,
    min_val_focus_positive: int = 10,
) -> dict[str, np.ndarray]:
    features: list[np.ndarray] = []
    dyaw: list[float] = []
    yaw_observable: list[float] = []
    label_valid: list[float] = []
    episode_idx: list[int] = []
    step_idx: list[int] = []
    keep_rows: list[int] = []
    yaw_entry_feasible: list[float] = []
    near_basin_shell: list[float] = []
    visual_observable: list[float] = []
    yaw_positive_focus: list[float] = []
    stratum: list[str] = []

    indexed_rows: list[tuple[int, dict[str, Any], float, float, float]] = []
    for idx, row in enumerate(rows):
        if stage_name and str(row.get("stage_name", "")) != stage_name:
            continue
        if skill_type and str(row.get("skill_type", "")) != skill_type:
            continue
        y, obs, valid = frame_yaw_label_from_row(row)
        if require_finite_label and (not np.isfinite(y) or valid < 0.5):
            continue
        indexed_rows.append((idx, row, float(y), float(obs), float(valid)))

    if balance_observability_pool:
        indexed_rows = _balance_observability_pool(
            indexed_rows,
            ratio=float(observability_negative_to_positive_ratio),
            seed=int(seed),
        )

    for idx, row, y, obs, valid in indexed_rows:
        features.append(frame_yaw_feature_vector(row))
        dyaw.append(float(y))
        yaw_observable.append(float(obs))
        label_valid.append(float(valid))
        episode_idx.append(int(row.get("episode_idx", -1)))
        step_idx.append(int(row.get("step_idx", row.get("step", -1))))
        keep_rows.append(int(idx))
        entry = _row_bool(row, "yaw_entry_feasible", False)
        near = _row_bool(row, "near_basin_shell", False)
        visual = _visual_class(row) == "visual_observable"
        yaw_entry_feasible.append(1.0 if entry else 0.0)
        near_basin_shell.append(1.0 if near else 0.0)
        visual_observable.append(1.0 if visual else 0.0)
        yaw_positive_focus.append(1.0 if (float(obs) > 0.5 and entry and near and visual) else 0.0)
        stratum.append(_yaw_stratum(row, float(obs)))

    if features:
        x = np.stack(features).astype(np.float32)
    else:
        x = np.zeros((0, len(FRAME_YAW_FEATURE_NAMES)), dtype=np.float32)
    episode_arr = np.asarray(episode_idx, dtype=np.int64)
    yaw_arr = np.asarray(yaw_observable, dtype=np.float32)
    focus_arr = np.asarray(yaw_positive_focus, dtype=np.float32)
    stratum_arr = np.asarray(stratum, dtype="<U32")
    split_arr = (
        _assign_balanced_split(
            episode_idx=episode_arr,
            yaw_observable=yaw_arr,
            yaw_positive_focus=focus_arr,
            stratum=stratum_arr,
            val_ratio=float(val_ratio),
            seed=int(seed),
            min_val_yaw_positive=int(min_val_yaw_positive),
            min_val_focus_positive=int(min_val_focus_positive),
        )
        if balanced_split
        else np.full((len(episode_idx),), "", dtype="<U8")
    )
    return {
        "features": x,
        "dyaw": np.asarray(dyaw, dtype=np.float32),
        "yaw_observable": yaw_arr,
        "label_valid": np.asarray(label_valid, dtype=np.float32),
        "episode_idx": episode_arr,
        "step_idx": np.asarray(step_idx, dtype=np.int64),
        "source_row_idx": np.asarray(keep_rows, dtype=np.int64),
        "yaw_entry_feasible": np.asarray(yaw_entry_feasible, dtype=np.float32),
        "near_basin_shell": np.asarray(near_basin_shell, dtype=np.float32),
        "visual_observable": np.asarray(visual_observable, dtype=np.float32),
        "yaw_positive_focus": focus_arr,
        "yaw_stratum": stratum_arr,
        "split": split_arr,
        "sample_weight": _balanced_sample_weights(yaw_arr, stratum_arr, focus_arr),
        "feature_names": np.asarray(list(FRAME_YAW_FEATURE_NAMES), dtype="<U64"),
        "observability_balanced_pool": np.asarray([1.0 if balance_observability_pool else 0.0], dtype=np.float32),
    }


def _split_summary(dataset: dict[str, np.ndarray]) -> dict[str, Any]:
    split = np.asarray(dataset.get("split", np.asarray([], dtype="<U8"))).astype(str)
    if split.size == 0 or not np.any(split != ""):
        return {}
    yaw = np.asarray(dataset["yaw_observable"], dtype=np.float32)
    focus = np.asarray(dataset["yaw_positive_focus"], dtype=np.float32)
    entry = np.asarray(dataset["yaw_entry_feasible"], dtype=np.float32)
    near = np.asarray(dataset["near_basin_shell"], dtype=np.float32)
    visual = np.asarray(dataset["visual_observable"], dtype=np.float32)
    ep = np.asarray(dataset["episode_idx"], dtype=np.int64)
    out: dict[str, Any] = {}
    for name in ("train", "val"):
        mask = split == name
        out[name] = {
            "rows": int(np.count_nonzero(mask)),
            "episodes": int(np.unique(ep[mask]).size) if np.any(mask) else 0,
            "yaw_positive_rows": int(np.count_nonzero(mask & (yaw > 0.5))),
            "yaw_positive_focus_rows": int(np.count_nonzero(mask & (focus > 0.5))),
            "yaw_entry_feasible_rows": int(np.count_nonzero(mask & (entry > 0.5))),
            "near_basin_shell_rows": int(np.count_nonzero(mask & (near > 0.5))),
            "visual_observable_rows": int(np.count_nonzero(mask & (visual > 0.5))),
        }
    return out


def write_dataset(
    dataset: dict[str, np.ndarray],
    output_npz: Path,
    *,
    source_jsonl: list[Path] | Path,
    stage_name: str,
    skill_type: str,
) -> Path:
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_npz, **dataset)
    dyaw = np.asarray(dataset["dyaw"], dtype=np.float32)
    obs = np.asarray(dataset["yaw_observable"], dtype=np.float32)
    report = {
        "schema_version": "frame_yaw_dataset_v1",
        "source_jsonl": [str(path.resolve()) for path in source_jsonl] if isinstance(source_jsonl, list) else str(source_jsonl.resolve()),
        "dataset_npz": str(output_npz.resolve()),
        "rows": int(dyaw.shape[0]),
        "feature_dim": int(dataset["features"].shape[1]),
        "feature_names": list(FRAME_YAW_FEATURE_NAMES),
        "stage_filter": str(stage_name),
        "skill_type_filter": str(skill_type),
        "episodes": int(np.unique(dataset["episode_idx"]).size) if dyaw.size else 0,
        "yaw_observable_rows": int(np.count_nonzero(obs > 0.5)),
        "yaw_observable_rate": float(np.mean(obs > 0.5)) if obs.size else 0.0,
        "yaw_positive_focus_rows": int(np.count_nonzero(np.asarray(dataset["yaw_positive_focus"], dtype=np.float32) > 0.5)),
        "yaw_entry_feasible_rows": int(np.count_nonzero(np.asarray(dataset["yaw_entry_feasible"], dtype=np.float32) > 0.5)),
        "near_basin_shell_rows": int(np.count_nonzero(np.asarray(dataset["near_basin_shell"], dtype=np.float32) > 0.5)),
        "visual_observable_rows": int(np.count_nonzero(np.asarray(dataset["visual_observable"], dtype=np.float32) > 0.5)),
        "stratum_counts": {
            str(name): int(np.count_nonzero(np.asarray(dataset["yaw_stratum"]).astype(str) == str(name)))
            for name in sorted(set(np.asarray(dataset["yaw_stratum"]).astype(str).tolist()))
        },
        "split_summary": _split_summary(dataset),
        "dyaw_abs_mean": float(np.mean(np.abs(dyaw))) if dyaw.size else 0.0,
        "dyaw_abs_p95": float(np.percentile(np.abs(dyaw), 95)) if dyaw.size else 0.0,
    }
    report_path = output_npz.with_suffix(".json")
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report_path


def main() -> None:
    ap = argparse.ArgumentParser(description="Build a C2C v2 frame yaw estimator dataset.")
    ap.add_argument("--relabel_jsonl", type=Path, nargs="+", required=True)
    ap.add_argument(
        "--output_npz",
        type=Path,
        default=Path("runtime_artifacts/coarse2contact_v2/datasets/frame_yaw_estimator_dataset.npz"),
    )
    ap.add_argument("--stage_name", type=str, default="RING_GRASP_ALIGN")
    ap.add_argument("--skill_type", type=str, default="precision_grasp")
    ap.add_argument("--allow_invalid_labels", action="store_true", default=False)
    ap.add_argument("--balanced_yaw_split", action="store_true", default=False)
    ap.add_argument("--balance_observability_pool", action="store_true", default=False)
    ap.add_argument("--observability_negative_to_positive_ratio", type=float, default=3.0)
    ap.add_argument("--val_ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--min_val_yaw_positive", type=int, default=30)
    ap.add_argument("--min_val_focus_positive", type=int, default=10)
    args = ap.parse_args()

    relabel_paths = _expand_inputs(list(args.relabel_jsonl))
    rows: list[dict[str, Any]] = []
    for path in relabel_paths:
        rows.extend(_read_jsonl(path))
    dataset = build_dataset(
        rows,
        stage_name=str(args.stage_name),
        skill_type=str(args.skill_type),
        require_finite_label=not bool(args.allow_invalid_labels),
        balanced_split=bool(args.balanced_yaw_split),
        balance_observability_pool=bool(args.balance_observability_pool),
        observability_negative_to_positive_ratio=float(args.observability_negative_to_positive_ratio),
        val_ratio=float(args.val_ratio),
        seed=int(args.seed),
        min_val_yaw_positive=int(args.min_val_yaw_positive),
        min_val_focus_positive=int(args.min_val_focus_positive),
    )
    report_path = write_dataset(dataset, args.output_npz, source_jsonl=relabel_paths, stage_name=str(args.stage_name), skill_type=str(args.skill_type))
    print(args.output_npz.resolve())
    print(report_path)


if __name__ == "__main__":
    main()
