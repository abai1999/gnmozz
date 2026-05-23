#!/usr/bin/env python3
"""Render one-page overlay and time-curve summaries for grasp-skill shadow traces."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def _load_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _font(size: int):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size=size)
    except Exception:
        return ImageFont.load_default()


def _crop_rgb(row: dict, crop_size: int = 96) -> Image.Image:
    img = Image.open(row["rgb_path"]).convert("RGB")
    x0, y0, x1, y1 = [int(v) for v in row.get("roi_box", [0, 0, img.size[0], img.size[1]])]
    crop = img.crop((x0, y0, x1, y1)).resize((crop_size, crop_size), resample=Image.BILINEAR)
    return crop


def _draw_marker(draw: ImageDraw.ImageDraw, x: float, y: float, color: tuple[int, int, int], r: int = 4) -> None:
    draw.ellipse((x - r, y - r, x + r, y + r), outline=color, width=2)


def _draw_thumb(row: dict, size: int = 96) -> Image.Image:
    img = _crop_rgb(row, crop_size=size)
    draw = ImageDraw.Draw(img)
    fw = float(row.get("frame_center_u", 0.5)) * float(size - 1)
    fh = float(row.get("frame_center_v", 0.5)) * float(size - 1)
    jw = float(row.get("jaw_reference_crop_u", 0.5)) * float(size - 1)
    jh = float(row.get("jaw_reference_crop_v", 0.5)) * float(size - 1)
    axis_dx = float(row.get("frame_axis_dir_x", 1.0)) * 18.0
    axis_dy = float(row.get("frame_axis_dir_y", 0.0)) * 18.0
    _draw_marker(draw, jw, jh, (255, 255, 255), r=4)
    _draw_marker(draw, fw, fh, (0, 255, 255), r=4)
    draw.line((fw - axis_dx, fh - axis_dy, fw + axis_dx, fh + axis_dy), fill=(255, 255, 0), width=2)
    txt = f"s2c={int(row.get('steps_to_close', -1))}"
    draw.text((4, 4), txt, fill=(255, 255, 255), font=_font(12))
    return img


def _panel_bounds(origin_x: int, origin_y: int, width: int, height: int) -> tuple[int, int, int, int]:
    return origin_x, origin_y, origin_x + width, origin_y + height


def _plot_series(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    xs: list[float],
    series: list[tuple[str, list[float], tuple[int, int, int]]],
    title: str,
    value_note: str,
) -> None:
    x0, y0, x1, y1 = box
    draw.rectangle(box, outline=(90, 90, 90), width=1)
    draw.text((x0 + 6, y0 + 4), f"{title}  {value_note}", fill=(235, 235, 235), font=_font(13))
    plot_left = x0 + 40
    plot_right = x1 - 8
    plot_top = y0 + 24
    plot_bottom = y1 - 18
    draw.line((plot_left, plot_bottom, plot_right, plot_bottom), fill=(110, 110, 110), width=1)
    draw.line((plot_left, plot_top, plot_left, plot_bottom), fill=(110, 110, 110), width=1)
    values = [float(v) for _, vals, _ in series for v in vals]
    if not values:
        return
    vabs = max(max(abs(v) for v in values), 1e-9)
    n = max(len(xs), 1)
    for label, vals, color in series:
        pts = []
        for i, v in enumerate(vals):
            px = plot_left if n == 1 else plot_left + (plot_right - plot_left) * float(i) / float(n - 1)
            py = 0.5 * (plot_top + plot_bottom) - (float(v) / vabs) * 0.46 * (plot_bottom - plot_top)
            pts.append((px, py))
        if len(pts) >= 2:
            draw.line(pts, fill=color, width=2)
        elif pts:
            _draw_marker(draw, pts[0][0], pts[0][1], color, r=3)
        draw.text((plot_right - 90, plot_top + 2 + 14 * series.index((label, vals, color))), label, fill=color, font=_font(12))
    draw.text((x0 + 4, plot_top), f"+{vabs:.4f}", fill=(180, 180, 180), font=_font(11))
    draw.text((x0 + 4, int(0.5 * (plot_top + plot_bottom)) - 6), "0", fill=(180, 180, 180), font=_font(11))
    draw.text((x0 + 4, plot_bottom - 10), f"-{vabs:.4f}", fill=(180, 180, 180), font=_font(11))
    for i, s in enumerate(xs):
        px = plot_left if n == 1 else plot_left + (plot_right - plot_left) * float(i) / float(n - 1)
        draw.text((px - 8, plot_bottom + 2), str(int(s)), fill=(180, 180, 180), font=_font(10))


def _render_episode(rows: list[dict], metrics: dict, out_path: Path) -> None:
    rows = sorted(rows, key=lambda r: int(r.get("step_idx", -1)))
    thumbs = [_draw_thumb(r, size=96) for r in rows]
    W, H = 1400, 900
    canvas = Image.new("RGB", (W, H), color=(18, 20, 24))
    draw = ImageDraw.Draw(canvas)
    title = (
        f"Episode {rows[0]['episode_idx']}  "
        f"xy rows={metrics.get('num_xyyaw_rows', 0)}  "
        f"pred/tgt dx corr={metrics.get('corr_pred_tgt_dx', float('nan')):.3f}  "
        f"dy corr={metrics.get('corr_pred_tgt_dy', float('nan')):.3f}  "
        f"dyaw corr={metrics.get('corr_pred_tgt_dyaw', float('nan')):.3f}  "
        f"active={metrics.get('pred_yaw_active_rate', float('nan')):.3f}  "
        f"stable={metrics.get('pred_yaw_active_stable_rate', float('nan')):.3f}  "
        f"geom={metrics.get('frame_geometry_consistent_rate', float('nan')):.3f}"
    )
    draw.text((20, 14), title, fill=(245, 245, 245), font=_font(20))
    draw.text(
        (20, 42),
        "thumb: white=jaw ref, cyan=ring center, yellow=ring axis | curves: red=pred, green=target, cyan=frame_rel (normalized for x/y, direct rad for yaw)",
        fill=(180, 185, 195),
        font=_font(13),
    )

    thumb_y = 80
    for i, thumb in enumerate(thumbs):
        canvas.paste(thumb, (20 + i * 102, thumb_y))

    xyyaw_rows = [r for r in rows if bool(r.get("xyyaw_supervision_mask", False))]
    if not xyyaw_rows:
        canvas.save(out_path)
        return
    xs = [int(r["steps_to_close"]) for r in xyyaw_rows]
    pred_dx = [float(r["pred_dx"]) for r in xyyaw_rows]
    pred_dy = [float(r["pred_dy"]) for r in xyyaw_rows]
    pred_dyaw = [float(r["pred_dyaw"]) for r in xyyaw_rows]
    tgt_dx = [float(r["tgt_dx"]) for r in xyyaw_rows]
    tgt_dy = [float(r["tgt_dy"]) for r in xyyaw_rows]
    tgt_dyaw = [float(r["tgt_dyaw"]) for r in xyyaw_rows]
    frame_rel_u = np.asarray([float(r["frame_rel_crop_u"]) for r in xyyaw_rows], dtype=np.float64)
    frame_rel_v = np.asarray([float(r["frame_rel_crop_v"]) for r in xyyaw_rows], dtype=np.float64)
    frame_rel_yaw = [float(r["frame_rel_yaw"]) for r in xyyaw_rows]

    def _normalize_like(ref: list[float], src: np.ndarray) -> list[float]:
        ref_abs = max(max(abs(v) for v in ref), 1e-9)
        src_abs = max(float(np.max(np.abs(src))), 1e-9)
        return (src / src_abs * ref_abs).tolist()

    frame_u_scaled = _normalize_like(tgt_dx, frame_rel_u)
    frame_v_scaled = _normalize_like(tgt_dy, frame_rel_v)

    panels = [
        ("dx", pred_dx, tgt_dx, frame_u_scaled, "m"),
        ("dy", pred_dy, tgt_dy, frame_v_scaled, "m"),
        ("dyaw", pred_dyaw, tgt_dyaw, frame_rel_yaw, "rad"),
    ]
    panel_y0 = 220
    panel_h = 190
    for idx, (name, pred, tgt, frame_rel, unit) in enumerate(panels):
        box = _panel_bounds(20, panel_y0 + idx * (panel_h + 18), 1360, panel_h)
        note = (
            f"pred std={np.std(np.asarray(pred, dtype=np.float64)):.4f}  "
            f"tgt std={np.std(np.asarray(tgt, dtype=np.float64)):.4f}"
        )
        _plot_series(
            draw,
            box,
            xs,
            [
                ("pred", pred, (255, 96, 96)),
                ("tgt", tgt, (128, 255, 128)),
                ("frame", frame_rel, (90, 215, 255)),
            ],
            title=name,
            value_note=note,
        )

    canvas.save(out_path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace_dir", type=Path, required=True)
    args = ap.parse_args()

    summary_path = args.trace_dir / "shadow_trace_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    render_dir = args.trace_dir / "plots"
    render_dir.mkdir(parents=True, exist_ok=True)

    rendered = []
    for ep_str, metrics in sorted(summary.get("episodes", {}).items(), key=lambda kv: int(kv[0])):
        trace_path = Path(metrics["trace_path"])
        rows = _load_rows(trace_path)
        out_path = render_dir / f"ep{int(ep_str):03d}_shadow_trace_plot.png"
        _render_episode(rows, metrics, out_path)
        rendered.append(out_path)

    if rendered:
        thumbs = []
        for p in rendered:
            img = Image.open(p).convert("RGB")
            thumbs.append(img.resize((420, 270), resample=Image.BILINEAR))
        cols = 2
        rows_n = int(math.ceil(len(thumbs) / cols))
        sheet = Image.new("RGB", (cols * 420, rows_n * 270), color=(0, 0, 0))
        for i, thumb in enumerate(thumbs):
            sheet.paste(thumb, ((i % cols) * 420, (i // cols) * 270))
        sheet_path = render_dir / "shadow_trace_plot_sheet.png"
        sheet.save(sheet_path)
    else:
        sheet_path = render_dir / "shadow_trace_plot_sheet.png"

    report = {
        "trace_dir": str(args.trace_dir),
        "plot_dir": str(render_dir),
        "sheet_path": str(sheet_path),
        "yaw_primary_metric_name": summary.get("yaw_primary_metric_name", "yaw_axis_cosine_mean"),
        "episodes": sorted(summary.get("episodes", {}).keys(), key=int),
    }
    (render_dir / "plot_report.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
