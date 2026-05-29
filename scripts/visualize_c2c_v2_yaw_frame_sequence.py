#!/usr/bin/env python3
"""Render a focused yaw/frame sequence visualization for a relabeled slice.

This utility is intentionally matplotlib-free so it can run in lightweight
environments.  It renders:

* an animated GIF over the selected step sequence,
* a compact timeline for raw / symmetry-aware / privileged yaw,
* a jump-sheet highlighting the largest symmetry-aware yaw flips,
* a JSON/MD report summarizing the slice.

The goal is diagnostic only.  It does not change runtime policy or relabels.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any
import sys

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size=size)
    except Exception:
        return ImageFont.load_default()


def _angular_diff(a: float, b: float) -> float:
    if not (np.isfinite(a) and np.isfinite(b)):
        return float("nan")
    return float(((a - b + math.pi) % (2.0 * math.pi)) - math.pi)


def _symmetry_aware(proxy_yaw: float, period: float) -> float:
    if not (np.isfinite(proxy_yaw) and np.isfinite(period) and period > 0.0):
        return float("nan")
    half = 0.5 * float(period)
    wrapped = ((float(proxy_yaw) + half) % float(period)) - half
    return float(-wrapped)


def _angle_to_vec(angle: float, radius: float) -> tuple[float, float]:
    return float(math.cos(angle) * radius), float(math.sin(angle) * radius)


def _draw_arrow(
    draw: ImageDraw.ImageDraw,
    center: tuple[float, float],
    angle: float,
    radius: float,
    color: tuple[int, int, int],
    width: int = 4,
    head: int = 10,
) -> None:
    if not np.isfinite(angle):
        return
    cx, cy = center
    dx, dy = _angle_to_vec(angle, radius)
    x2 = cx + dx
    y2 = cy - dy
    draw.line((cx, cy, x2, y2), fill=color, width=width)
    head_left = angle + math.pi * 0.82
    head_right = angle - math.pi * 0.82
    for h in (head_left, head_right):
        hx, hy = _angle_to_vec(h, head)
        draw.line((x2, y2, x2 + hx, y2 - hy), fill=color, width=max(1, width - 1))


def _draw_circle_gauge(
    canvas: Image.Image,
    box: tuple[int, int, int, int],
    raw_yaw: float,
    symm_yaw: float,
    priv_yaw: float,
    title: str,
) -> None:
    draw = ImageDraw.Draw(canvas)
    x0, y0, x1, y1 = box
    draw.rectangle(box, outline=(90, 90, 90), width=1)
    draw.text((x0 + 8, y0 + 6), title, fill=(235, 235, 235), font=_font(14))
    cx = 0.5 * (x0 + x1)
    cy = 0.5 * (y0 + y1) + 8
    radius = min(x1 - x0, y1 - y0) * 0.32
    draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), outline=(155, 155, 155), width=2)
    for ang in (0.0, math.pi / 2.0, math.pi, -math.pi / 2.0):
        dx, dy = _angle_to_vec(ang, radius)
        draw.line((cx, cy, cx + dx, cy - dy), fill=(70, 70, 70), width=1)
    _draw_arrow(draw, (cx, cy), raw_yaw, radius * 0.92, (255, 92, 92), width=4, head=9)
    _draw_arrow(draw, (cx, cy), symm_yaw, radius * 0.74, (92, 255, 140), width=4, head=9)
    _draw_arrow(draw, (cx, cy), priv_yaw, radius * 0.56, (100, 180, 255), width=4, head=8)
    draw.text((x0 + 10, y1 - 56), "raw  red", fill=(255, 92, 92), font=_font(13))
    draw.text((x0 + 10, y1 - 40), "symm green", fill=(92, 255, 140), font=_font(13))
    draw.text((x0 + 10, y1 - 24), "priv  blue", fill=(100, 180, 255), font=_font(13))


def _normalize_depth(depth: np.ndarray) -> Image.Image:
    arr = np.asarray(depth, dtype=np.float32)
    finite = np.isfinite(arr)
    if not np.any(finite):
        return Image.new("L", (arr.shape[1], arr.shape[0]), color=0)
    vals = arr[finite]
    lo = float(np.percentile(vals, 5.0))
    hi = float(np.percentile(vals, 95.0))
    if hi <= lo + 1e-8:
        hi = lo + 1e-8
    clipped = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
    out = (255.0 * (1.0 - clipped)).astype(np.uint8)
    return Image.fromarray(out, mode="L")


def _draw_series_panel(
    canvas: Image.Image,
    box: tuple[int, int, int, int],
    steps: list[int],
    raw: list[float],
    symm: list[float],
    priv: list[float],
    obs: list[float],
    jump_steps: set[int],
    current_step: int,
    title: str,
) -> None:
    draw = ImageDraw.Draw(canvas)
    x0, y0, x1, y1 = box
    draw.rectangle(box, outline=(90, 90, 90), width=1)
    draw.text((x0 + 8, y0 + 6), title, fill=(235, 235, 235), font=_font(14))
    plot_left = x0 + 58
    plot_right = x1 - 16
    plot_top = y0 + 28
    plot_bottom = y1 - 20
    draw.line((plot_left, plot_bottom, plot_right, plot_bottom), fill=(110, 110, 110), width=1)
    draw.line((plot_left, plot_top, plot_left, plot_bottom), fill=(110, 110, 110), width=1)

    all_vals = [v for v in raw + symm + priv if np.isfinite(v)]
    if not all_vals:
        return
    vabs = max(max(abs(v) for v in all_vals), 1e-6)
    n = max(len(steps), 1)
    x_min = min(steps) if steps else 0
    x_max = max(steps) if steps else 1
    if x_max <= x_min:
        x_max = x_min + 1

    def _xy(idx: int, val: float) -> tuple[float, float]:
        step = steps[idx]
        px = plot_left + (plot_right - plot_left) * float(step - x_min) / float(x_max - x_min)
        py = 0.5 * (plot_top + plot_bottom) - (float(val) / vabs) * 0.44 * (plot_bottom - plot_top)
        return px, py

    def _series(vals: list[float], color: tuple[int, int, int], label: str, y_offset: int) -> None:
        pts = [(_xy(i, v) if np.isfinite(v) else None) for i, v in enumerate(vals)]
        clean_pts = [p for p in pts if p is not None]
        if len(clean_pts) >= 2:
            draw.line(clean_pts, fill=color, width=2)
        for i, p in enumerate(pts):
            if p is None:
                continue
            px, py = p
            r = 3 if label != "priv" else 4
            draw.ellipse((px - r, py - r, px + r, py + r), outline=color, width=2)
        draw.text((plot_right - 120, plot_top + y_offset), label, fill=color, font=_font(12))

    _series(raw, (255, 92, 92), "raw", 0)
    _series(symm, (92, 255, 140), "symm", 14)
    _series(priv, (100, 180, 255), "priv", 28)

    # Frame observability as a thin gray trace, scaled to the panel height.
    obs_vals = [v for v in obs if np.isfinite(v)]
    if obs_vals:
        obs_max = max(max(obs_vals), 1e-6)
        pts = []
        for i, v in enumerate(obs):
            if not np.isfinite(v):
                continue
            px, _ = _xy(i, 0.0)
            py = plot_bottom - (float(v) / obs_max) * (plot_bottom - plot_top) * 0.60
            pts.append((px, py))
        if len(pts) >= 2:
            draw.line(pts, fill=(180, 180, 180), width=1)
        draw.text((plot_left + 4, plot_top + 2), f"frame_obs max={obs_max:.4f}", fill=(180, 180, 180), font=_font(11))

    # Current step and jump markers.
    for i, step in enumerate(steps):
        px, _ = _xy(i, 0.0)
        if step in jump_steps:
            draw.line((px, plot_top, px, plot_bottom), fill=(255, 170, 60), width=1)
            draw.text((px - 8, plot_top - 12), "!", fill=(255, 170, 60), font=_font(11))
        if step == current_step:
            draw.line((px, plot_top, px, plot_bottom), fill=(255, 255, 255), width=2)
            draw.text((px - 10, plot_bottom + 2), str(step), fill=(255, 255, 255), font=_font(10))
        else:
            stride = max(1, len(steps) // 12)
            if i % stride == 0:
                draw.text((px - 8, plot_bottom + 2), str(step), fill=(170, 170, 170), font=_font(10))

    draw.text((plot_left, plot_bottom + 14), f"y-range +/-{vabs:.3f} rad", fill=(170, 170, 170), font=_font(11))


def _render_frame(
    rgb: np.ndarray,
    depth: np.ndarray,
    row: dict[str, Any],
    *,
    raw_yaw: float,
    symm_yaw: float,
    priv_yaw: float,
    bias_corrected: float,
    jump_flag: bool,
    jump_delta: float,
    steps: list[int],
    raw_series: list[float],
    symm_series: list[float],
    priv_series: list[float],
    obs_series: list[float],
    jump_steps: set[int],
) -> Image.Image:
    base = Image.fromarray(np.asarray(rgb, dtype=np.uint8), mode="RGB").resize((420, 420), resample=Image.BILINEAR)
    depth_img = _normalize_depth(depth).convert("RGB").resize((220, 220), resample=Image.NEAREST)
    depth_img = depth_img.filter(ImageFilter.SHARPEN)
    edge_img = Image.fromarray(np.asarray(rgb, dtype=np.uint8), mode="RGB").convert("L").filter(ImageFilter.FIND_EDGES).convert("RGB").resize((220, 220), resample=Image.BILINEAR)

    canvas = Image.new("RGB", (1220, 860), color=(18, 20, 24))
    draw = ImageDraw.Draw(canvas)

    # Top title.
    ep = int(row.get("episode_idx", -1))
    step = int(row.get("step_idx", -1))
    obs = _safe_float(row.get("source_frame_observability", row.get("yaw_observability_frame_observability", float("nan"))))
    conf = _safe_float(row.get("source_frame_confidence", row.get("yaw_observability_frame_confidence", float("nan"))))
    bucket = str(row.get("failure_bucket", ""))
    blocker = str(row.get("yaw_observability_primary_blocker", ""))
    title = (
        f"ep{ep:03d} step {step}  bucket={bucket}  blocker={blocker}  "
        f"obs={obs:.4f} conf={conf:.3f}  jump={'yes' if jump_flag else 'no'}"
    )
    draw.text((20, 14), title, fill=(245, 245, 245), font=_font(20))
    draw.text(
        (20, 42),
        "red=raw proxy yaw  green=symmetry-aware baseline  blue=privileged truth  gray=line=frame observability",
        fill=(180, 185, 195),
        font=_font(13),
    )

    # Left image and insets.
    canvas.paste(base, (20, 80))
    draw.rectangle((20, 80, 440, 500), outline=(90, 90, 90), width=1)
    draw.text((26, 486), "wrist_rgb", fill=(220, 220, 220), font=_font(12))

    # Add a contour proxy inset using RGB edge response.
    canvas.paste(edge_img, (460, 80))
    draw.rectangle((460, 80, 680, 300), outline=(90, 90, 90), width=1)
    draw.text((466, 286), "rgb edge proxy", fill=(220, 220, 220), font=_font(12))

    canvas.paste(depth_img, (460, 320))
    draw.rectangle((460, 320, 680, 540), outline=(90, 90, 90), width=1)
    draw.text((466, 526), "wrist_depth", fill=(220, 220, 220), font=_font(12))

    # Gauge and numeric summary.
    _draw_circle_gauge(
        canvas,
        (700, 80, 1200, 340),
        raw_yaw=raw_yaw,
        symm_yaw=symm_yaw,
        priv_yaw=priv_yaw,
        title="yaw gauge",
    )
    metrics_box = (700, 360, 1200, 540)
    draw.rectangle(metrics_box, outline=(90, 90, 90), width=1)
    draw.text((710, 370), f"raw proxy yaw: {raw_yaw:+.4f} rad", fill=(255, 92, 92), font=_font(16))
    draw.text((710, 398), f"symmetry-aware: {symm_yaw:+.4f} rad", fill=(92, 255, 140), font=_font(16))
    draw.text((710, 426), f"privileged truth: {priv_yaw:+.4f} rad", fill=(100, 180, 255), font=_font(16))
    draw.text((710, 454), f"bias-corrected symm: {bias_corrected:+.4f} rad", fill=(235, 235, 235), font=_font(16))
    draw.text((710, 482), f"jump delta from prev: {jump_delta:+.4f} rad", fill=(255, 170, 60), font=_font(16))
    draw.text((710, 510), f"frame_observability_lt_010={'yes' if obs < 0.010 else 'no'}", fill=(220, 220, 220), font=_font(16))

    # Timeline panel.
    _draw_series_panel(
        canvas,
        (20, 540, 1200, 820),
        steps=steps,
        raw=raw_series,
        symm=symm_series,
        priv=priv_series,
        obs=obs_series,
        jump_steps=jump_steps,
        current_step=step,
        title="step sequence",
    )
    return canvas


def _load_slice(
    relabel_jsonl: Path,
    episode_idx: int,
    failure_bucket: str,
    primary_blocker: str,
    stage_name: str,
    skill_type: str,
) -> list[dict[str, Any]]:
    rows = _read_jsonl(relabel_jsonl)
    selected: list[dict[str, Any]] = []
    for row in rows:
        if int(row.get("episode_idx", -1)) != int(episode_idx):
            continue
        if stage_name and str(row.get("stage_name", "")) != stage_name:
            continue
        if skill_type and str(row.get("skill_type", "")) != skill_type:
            continue
        if failure_bucket and str(row.get("failure_bucket", "")) != failure_bucket:
            continue
        if primary_blocker and str(row.get("yaw_observability_primary_blocker", "")) != primary_blocker:
            continue
        selected.append(row)
    selected.sort(key=lambda r: int(r.get("step_idx", -1)))
    return selected


def _episode_arrays(npz_path: Path) -> dict[str, np.ndarray]:
    arr = np.load(npz_path, allow_pickle=True)
    return {k: arr[k] for k in arr.files}


def _step_to_index(step_array: np.ndarray) -> dict[int, int]:
    out: dict[int, int] = {}
    for idx, step in enumerate(np.asarray(step_array).reshape(-1).tolist()):
        try:
            out[int(step)] = int(idx)
        except Exception:
            continue
    return out


def _load_rgb_depth_from_image_dir(image_dir: Path, step: int) -> tuple[np.ndarray, np.ndarray]:
    rgb_path = image_dir / f"step{int(step):04d}_rgb.png"
    depth_path = image_dir / f"step{int(step):04d}_depth.png"
    if not rgb_path.exists() or not depth_path.exists():
        raise FileNotFoundError(f"Missing image pair for step {step}: {rgb_path} / {depth_path}")
    rgb = np.asarray(Image.open(rgb_path).convert("RGB"))
    depth = np.asarray(Image.open(depth_path).convert("L"))
    return rgb, depth


def _available_image_steps(image_dir: Path) -> set[int]:
    steps: set[int] = set()
    for path in image_dir.glob("step*_rgb.png"):
        name = path.stem
        if not name.startswith("step") or not name.endswith("_rgb"):
            continue
        step_str = name[len("step") : -len("_rgb")]
        try:
            steps.add(int(step_str))
        except Exception:
            continue
    return steps


def _render_jump_sheet(frames: list[tuple[Image.Image, str]], out_path: Path, cols: int = 2) -> None:
    if not frames:
        return
    w, h = frames[0][0].size
    label_h = 30
    tile_h = h + label_h
    rows = int(math.ceil(len(frames) / cols))
    sheet = Image.new("RGB", (cols * w, rows * tile_h), color=(14, 16, 20))
    draw = ImageDraw.Draw(sheet)
    for idx, (img, label) in enumerate(frames):
        x = (idx % cols) * w
        y = (idx // cols) * tile_h
        sheet.paste(img, (x, y + label_h))
        draw.rectangle((x, y, x + w - 1, y + tile_h - 1), outline=(80, 80, 80), width=1)
        draw.text((x + 8, y + 6), label, fill=(235, 235, 235), font=_font(14))
    sheet.save(out_path)


def main() -> None:
    ap = argparse.ArgumentParser(description="Render a focused yaw/frame diagnostic sequence for one ep/bucket slice.")
    ap.add_argument(
        "--relabel_jsonl",
        type=Path,
        default=Path("runtime_artifacts/coarse2contact_v2/relabels/planner_only_30ep_hardmix_frame_residual_v2/frame_residual_v2.jsonl"),
    )
    ap.add_argument(
        "--runtime_obs_npz",
        type=Path,
        default=Path("runtime_artifacts/coarse2contact_v2/runtime_failure_traces/planner_only_30ep_hardmix/runtime_observations/ep006_runtime_obs.npz"),
    )
    ap.add_argument(
        "--image_dir",
        type=Path,
        default=None,
        help="Optional directory containing stepXXXX_rgb.png and stepXXXX_depth.png files.",
    )
    ap.add_argument("--episode_idx", type=int, default=6)
    ap.add_argument("--failure_bucket", type=str, default="small_xy_small_yaw")
    ap.add_argument("--primary_blocker", type=str, default="frame_observability_lt_010")
    ap.add_argument("--stage_name", type=str, default="RING_GRASP_ALIGN")
    ap.add_argument("--skill_type", type=str, default="precision_grasp")
    ap.add_argument(
        "--output_dir",
        type=Path,
        default=Path("runtime_artifacts/coarse2contact_v2/reports/yaw_frame_alignment_diagnostic/ep006_small_xy_small_yaw_frame_observability_lt010"),
    )
    ap.add_argument("--jump_threshold", type=float, default=0.40)
    ap.add_argument("--fps", type=int, default=3)
    args = ap.parse_args()

    rows = _load_slice(
        args.relabel_jsonl,
        episode_idx=int(args.episode_idx),
        failure_bucket=str(args.failure_bucket),
        primary_blocker=str(args.primary_blocker),
        stage_name=str(args.stage_name),
        skill_type=str(args.skill_type),
    )
    if not rows:
        raise RuntimeError("No rows matched the requested slice")

    arrays: dict[str, np.ndarray] | None = None
    step_to_idx: dict[int, int] | None = None
    if args.image_dir is None:
        arrays = _episode_arrays(args.runtime_obs_npz)
        step_to_idx = _step_to_index(arrays["step"])
        if int(rows[0]["step_idx"]) not in step_to_idx:
            raise RuntimeError("The runtime observation step index does not align with the relabel steps")
    else:
        if not args.image_dir.exists():
            raise FileNotFoundError(f"image_dir does not exist: {args.image_dir}")
        available_steps = _available_image_steps(args.image_dir)
        rows = [row for row in rows if int(row["step_idx"]) in available_steps]
        if not rows:
            raise RuntimeError(f"No relabel rows have image pairs in {args.image_dir}")
        print(
            json.dumps(
                {
                    "image_dir": str(args.image_dir.resolve()),
                    "available_steps": len(available_steps),
                    "rendered_rows": len(rows),
                    "first_step": int(rows[0]["step_idx"]),
                    "last_step": int(rows[-1]["step_idx"]),
                },
                sort_keys=True,
            )
        )

    steps = [int(r["step_idx"]) for r in rows]
    raw_series: list[float] = []
    symm_series: list[float] = []
    priv_series: list[float] = []
    obs_series: list[float] = []
    bias_series: list[float] = []
    jump_steps: set[int] = set()
    jump_info: list[dict[str, Any]] = []
    period = float(math.pi / 2.0)

    for row in rows:
        proxy = row.get("proxy_local_geometry_error", {})
        raw = _safe_float(proxy.get("dyaw", proxy.get("image_axis_yaw", float("nan"))))
        priv = _safe_float(row.get("privileged_dyaw", row.get("true_basin_error_t", {}).get("dyaw", float("nan"))))
        symm = _symmetry_aware(raw, period)
        obs = _safe_float(row.get("source_frame_observability", row.get("yaw_observability_frame_observability", float("nan"))))
        raw_series.append(raw)
        symm_series.append(symm)
        priv_series.append(priv)
        obs_series.append(obs)

    mask = np.isfinite(np.asarray(symm_series, dtype=np.float64)) & np.isfinite(np.asarray(priv_series, dtype=np.float64))
    bias = float(np.mean((np.asarray(symm_series, dtype=np.float64)[mask] - np.asarray(priv_series, dtype=np.float64)[mask]))) if np.any(mask) else 0.0
    bias_series = [float(v - bias) if np.isfinite(v) else float("nan") for v in symm_series]

    for i in range(1, len(rows)):
        d = abs(_angular_diff(symm_series[i], symm_series[i - 1]))
        if np.isfinite(d) and d >= float(args.jump_threshold):
            jump_steps.add(int(rows[i]["step_idx"]))
            jump_info.append(
                {
                    "prev_step": int(rows[i - 1]["step_idx"]),
                    "step": int(rows[i]["step_idx"]),
                    "delta_symm": float(d),
                    "prev_obs": float(obs_series[i - 1]),
                    "obs": float(obs_series[i]),
                    "prev_raw": float(raw_series[i - 1]),
                    "raw": float(raw_series[i]),
                    "prev_symm": float(symm_series[i - 1]),
                    "symm": float(symm_series[i]),
                    "prev_priv": float(priv_series[i - 1]),
                    "priv": float(priv_series[i]),
                }
            )

    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    frames: list[Image.Image] = []
    for row, raw, symm, priv, bias_corr, obs in zip(rows, raw_series, symm_series, priv_series, bias_series, obs_series):
        step = int(row["step_idx"])
        if arrays is not None and step_to_idx is not None:
            idx = step_to_idx[step]
            rgb = arrays["wrist_rgb"][idx]
            depth = arrays["wrist_depth"][idx]
        else:
            rgb, depth = _load_rgb_depth_from_image_dir(args.image_dir, step)
        jump_delta = 0.0
        if len(frames) > 0:
            jump_delta = abs(_angular_diff(symm, symm_series[len(frames) - 1]))
        jump_flag = step in jump_steps
        frame = _render_frame(
            rgb=rgb,
            depth=depth,
            row=row,
            raw_yaw=raw,
            symm_yaw=symm,
            priv_yaw=priv,
            bias_corrected=bias_corr,
            jump_flag=jump_flag,
            jump_delta=jump_delta,
            steps=steps,
            raw_series=raw_series,
            symm_series=symm_series,
            priv_series=priv_series,
            obs_series=obs_series,
            jump_steps=jump_steps,
        )
        frames.append(frame)

    gif_path = plot_dir / f"ep{int(args.episode_idx):03d}_{args.failure_bucket}_{args.primary_blocker}.gif"
    imageio.mimsave(gif_path, [np.asarray(frame) for frame in frames], fps=max(1, int(args.fps)))

    # Key jump sheet: previous/current frames for the largest flips.
    ranked = sorted(jump_info, key=lambda item: item["delta_symm"], reverse=True)
    jump_tiles: list[tuple[Image.Image, str]] = []
    for item in ranked[:8]:
        prev_step = int(item["prev_step"])
        step = int(item["step"])
        if arrays is not None and step_to_idx is not None:
            prev_idx = step_to_idx[prev_step]
            curr_idx = step_to_idx[step]
            prev_rgb = arrays["wrist_rgb"][prev_idx]
            prev_depth = arrays["wrist_depth"][prev_idx]
            curr_rgb = arrays["wrist_rgb"][curr_idx]
            curr_depth = arrays["wrist_depth"][curr_idx]
        else:
            prev_rgb, prev_depth = _load_rgb_depth_from_image_dir(args.image_dir, prev_step)
            curr_rgb, curr_depth = _load_rgb_depth_from_image_dir(args.image_dir, step)
        prev_frame = _render_frame(
            rgb=prev_rgb,
            depth=prev_depth,
            row=rows[steps.index(prev_step)],
            raw_yaw=float(raw_series[steps.index(prev_step)]),
            symm_yaw=float(symm_series[steps.index(prev_step)]),
            priv_yaw=float(priv_series[steps.index(prev_step)]),
            bias_corrected=float(bias_series[steps.index(prev_step)]),
            jump_flag=False,
            jump_delta=0.0,
            steps=steps,
            raw_series=raw_series,
            symm_series=symm_series,
            priv_series=priv_series,
            obs_series=obs_series,
            jump_steps=jump_steps,
        )
        curr_frame = _render_frame(
            rgb=curr_rgb,
            depth=curr_depth,
            row=rows[steps.index(step)],
            raw_yaw=float(raw_series[steps.index(step)]),
            symm_yaw=float(symm_series[steps.index(step)]),
            priv_yaw=float(priv_series[steps.index(step)]),
            bias_corrected=float(bias_series[steps.index(step)]),
            jump_flag=True,
            jump_delta=float(item["delta_symm"]),
            steps=steps,
            raw_series=raw_series,
            symm_series=symm_series,
            priv_series=priv_series,
            obs_series=obs_series,
            jump_steps=jump_steps,
        )
        pair = Image.new("RGB", (prev_frame.size[0] * 2 + 20, prev_frame.size[1]), color=(14, 16, 20))
        pair.paste(prev_frame, (0, 0))
        pair.paste(curr_frame, (prev_frame.size[0] + 20, 0))
        label = (
            f"step {prev_step} -> {step}  "
            f"delta_symm={item['delta_symm']:.3f}  "
            f"obs {item['prev_obs']:.4f}->{item['obs']:.4f}"
        )
        jump_tiles.append((pair, label))

    jump_sheet = plot_dir / f"ep{int(args.episode_idx):03d}_{args.failure_bucket}_{args.primary_blocker}_jump_sheet.png"
    _render_jump_sheet(jump_tiles, jump_sheet, cols=1 if len(jump_tiles) <= 2 else 2)

    # Compact summary report.
    proxy_arr = np.asarray(raw_series, dtype=np.float64)
    priv_arr = np.asarray(priv_series, dtype=np.float64)
    symm_arr = np.asarray(symm_series, dtype=np.float64)
    mask = np.isfinite(proxy_arr) & np.isfinite(priv_arr)
    raw_mae = float(np.mean(np.abs(proxy_arr[mask] - priv_arr[mask]))) if np.any(mask) else 0.0
    symm_mae = float(np.mean(np.abs(symm_arr[mask] - priv_arr[mask]))) if np.any(mask) else 0.0
    bias_corr_mae = float(np.mean(np.abs((symm_arr[mask] - bias) - priv_arr[mask]))) if np.any(mask) else 0.0
    summary = {
        "source_jsonl": str(args.relabel_jsonl.resolve()),
        "source_npz": None if args.image_dir is not None else str(args.runtime_obs_npz.resolve()),
        "source_image_dir": None if args.image_dir is None else str(args.image_dir.resolve()),
        "episode_idx": int(args.episode_idx),
        "failure_bucket": str(args.failure_bucket),
        "primary_blocker": str(args.primary_blocker),
        "stage_name": str(args.stage_name),
        "skill_type": str(args.skill_type),
        "num_rows": int(len(rows)),
        "selected_step_idxs": [int(row["step_idx"]) for row in rows],
        "step_idx_min": int(min(int(row["step_idx"]) for row in rows)),
        "step_idx_max": int(max(int(row["step_idx"]) for row in rows)),
        "num_jump_points": int(len(jump_info)),
        "raw_proxy_mae": raw_mae,
        "symmetry_aware_mae": symm_mae,
        "symmetry_aware_bias": bias,
        "bias_corrected_mae": bias_corr_mae,
        "gif_path": str(gif_path),
        "jump_sheet_path": str(jump_sheet),
        "jump_points": jump_info[:20],
    }
    (out_dir / "yaw_frame_sequence_report.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    md_lines = [
        "# Yaw / Frame Sequence Visualization",
        "",
        f"- source relabel: `{args.relabel_jsonl}`",
        f"- source npz: `{args.runtime_obs_npz}`" if args.image_dir is None else f"- source image_dir: `{args.image_dir}`",
        f"- slice: `ep{int(args.episode_idx):03d}` / `{args.failure_bucket}` / `{args.primary_blocker}`",
        f"- rows: `{len(rows)}`",
        f"- jump points: `{len(jump_info)}`",
        f"- raw proxy MAE: `{raw_mae:.6f}`",
        f"- symmetry-aware MAE: `{symm_mae:.6f}`",
        f"- symmetry-aware bias: `{bias:.6f}`",
        f"- bias-corrected MAE: `{bias_corr_mae:.6f}`",
        "",
        "## Jump Points",
    ]
    for item in jump_info[:20]:
        md_lines.append(
            f"- step {item['prev_step']} -> {item['step']}: d_symm={item['delta_symm']:.3f}, "
            f"obs {item['prev_obs']:.4f}->{item['obs']:.4f}, raw {item['prev_raw']:.3f}->{item['raw']:.3f}, "
            f"symm {item['prev_symm']:.3f}->{item['symm']:.3f}, priv {item['prev_priv']:.3f}->{item['priv']:.3f}"
        )
    md_lines.extend(
        [
            "",
            "## Outputs",
            f"- GIF: `{gif_path}`",
            f"- Jump sheet: `{jump_sheet}`",
        ]
    )
    (out_dir / "yaw_frame_sequence_report.md").write_text("\n".join(md_lines), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
