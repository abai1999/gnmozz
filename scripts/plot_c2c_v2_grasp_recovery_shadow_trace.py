#!/usr/bin/env python3
"""Render recovery shadow trajectory plots from per-step trace rows."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
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


def _draw_panel(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], title: str, xs: list[int], series: list[tuple[str, list[float], tuple[int, int, int]]], note: str) -> None:
    x0, y0, x1, y1 = box
    draw.rectangle(box, outline=(90, 90, 90), width=1)
    draw.text((x0 + 6, y0 + 4), title, fill=(245, 245, 245), font=_font(13))
    draw.text((x0 + 6, y0 + 20), note, fill=(185, 190, 200), font=_font(11))
    left = x0 + 38
    right = x1 - 8
    top = y0 + 38
    bottom = y1 - 20
    draw.line((left, bottom, right, bottom), fill=(110, 110, 110), width=1)
    draw.line((left, top, left, bottom), fill=(110, 110, 110), width=1)
    vals = [v for _, s, _ in series for v in s]
    if not vals:
        return
    vmax = max(max(abs(float(v)) for v in vals), 1e-9)
    n = max(len(xs), 1)
    for idx, (label, vals, color) in enumerate(series):
        pts = []
        for i, v in enumerate(vals):
            px = left if n == 1 else left + (right - left) * float(i) / float(n - 1)
            py = 0.5 * (top + bottom) - (float(v) / vmax) * 0.45 * (bottom - top)
            pts.append((px, py))
        if len(pts) >= 2:
            draw.line(pts, fill=color, width=2)
        elif pts:
            draw.ellipse((pts[0][0] - 2, pts[0][1] - 2, pts[0][0] + 2, pts[0][1] + 2), outline=color, width=1)
        draw.text((right - 92, top + 2 + 14 * idx), label, fill=color, font=_font(11))
    draw.text((x0 + 4, top), f"+{vmax:.4f}", fill=(180, 180, 180), font=_font(10))
    draw.text((x0 + 4, int(0.5 * (top + bottom)) - 6), "0", fill=(180, 180, 180), font=_font(10))
    draw.text((x0 + 4, bottom - 10), f"-{vmax:.4f}", fill=(180, 180, 180), font=_font(10))
    for i, step in enumerate(xs):
        px = left if n == 1 else left + (right - left) * float(i) / float(n - 1)
        draw.text((px - 8, bottom + 2), str(int(step)), fill=(170, 170, 170), font=_font(9))


def _render_episode(rows: list[dict], out_path: Path) -> None:
    rows = sorted(rows, key=lambda r: int(r.get("trajectory_step", r.get("step_idx", 0))))
    xs = [int(r.get("trajectory_step", r.get("step_idx", 0))) for r in rows]
    err_norm = [float(r.get("recovery_error_norm", 0.0)) for r in rows]
    post_norm = [float(r.get("recovery_post_norm", 0.0)) for r in rows]
    gain = [float(r.get("recovery_gain", 0.0)) for r in rows]
    bias = [float(r.get("planner_bias_score", 0.0)) for r in rows]
    pred_xy = [float(r.get("pred_norm_xy", 0.0)) for r in rows]
    tgt_xy = [float(r.get("error_norm_xy", 0.0)) for r in rows]
    pred_yaw = [float(r.get("pred_abs_yaw", 0.0)) for r in rows]
    tgt_yaw = [float(r.get("error_abs_yaw", 0.0)) for r in rows]
    canvas = Image.new("RGB", (1400, 780), color=(18, 20, 24))
    draw = ImageDraw.Draw(canvas)
    header = (
        f"Episode {rows[0]['episode_idx']}  "
        f"gain_mean={np.mean(gain):.5f}  "
        f"improved_rate={np.mean(np.asarray(gain) > 0.0):.3f}  "
        f"bias_start={bias[0]:.6f}  bias_end={bias[-1]:.6f}"
    )
    draw.text((18, 14), header, fill=(245, 245, 245), font=_font(18))
    draw.text((18, 40), "red=pred, green=target, cyan=post", fill=(185, 190, 200), font=_font(12))
    panels = [
        ("recovery_error_norm", err_norm, post_norm, gain),
        ("planner_bias_score", bias, [bias[0]] * len(bias), gain),
        ("xy_norm", pred_xy, tgt_xy, gain),
        ("yaw_abs", pred_yaw, tgt_yaw, gain),
    ]
    for i, (name, pred, tgt, aux) in enumerate(panels):
        box = (18 + (i % 2) * 680, 80 + (i // 2) * 330, 18 + (i % 2) * 680 + 660, 80 + (i // 2) * 330 + 300)
        _draw_panel(
            draw,
            box,
            name,
            xs,
            [
                ("pred", pred, (255, 96, 96)),
                ("tgt", tgt, (128, 255, 128)),
                ("gain", aux, (90, 215, 255)),
            ],
            note=f"start={pred[0]:.5f} end={pred[-1]:.5f}",
        )
    canvas.save(out_path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace_output", type=Path, required=True)
    ap.add_argument("--output_dir", type=Path, required=True)
    ap.add_argument("--top_k", type=int, default=8)
    args = ap.parse_args()

    rows = _load_rows(args.trace_output)
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("trajectory_id", f"ep{int(row.get('episode_idx', -1)):03d}"))].append(row)

    traj_summaries = []
    for traj_id, traj in grouped.items():
        traj = sorted(traj, key=lambda r: int(r.get("trajectory_step", r.get("step_idx", 0))))
        gains = np.asarray([float(r.get("recovery_gain", 0.0)) for r in traj], dtype=np.float32)
        traj_summaries.append(
            {
                "trajectory_id": traj_id,
                "episode_idx": int(traj[0].get("episode_idx", -1)),
                "gain_mean": float(np.mean(gains)),
                "gain_rate": float(np.mean(gains > 0.0)),
                "bias_score_max": float(np.max([float(r.get("planner_bias_score", 0.0)) for r in traj])),
                "error_norm_start": float(traj[0].get("recovery_error_norm", 0.0)),
                "error_norm_end": float(traj[-1].get("recovery_error_norm", 0.0)),
                "trajectory_improved": bool(float(traj[-1].get("recovery_error_norm", 0.0)) < float(traj[0].get("recovery_error_norm", 0.0))),
            }
        )

    hard = sorted(traj_summaries, key=lambda r: (r["gain_mean"], -r["bias_score_max"]))[: max(1, int(args.top_k))]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rendered = []
    for item in hard:
        traj_id = item["trajectory_id"]
        out_path = args.output_dir / f"{traj_id}_recovery_shadow_plot.png"
        _render_episode(grouped[traj_id], out_path)
        rendered.append(out_path)

    if rendered:
        thumbs = [Image.open(p).convert("RGB").resize((420, 235), resample=Image.BILINEAR) for p in rendered]
        cols = 2
        rows_n = int(math.ceil(len(thumbs) / cols))
        sheet = Image.new("RGB", (cols * 420, rows_n * 235), color=(8, 10, 14))
        for idx, thumb in enumerate(thumbs):
            sheet.paste(thumb, ((idx % cols) * 420, (idx // cols) * 235))
        sheet_path = args.output_dir / "recovery_shadow_plot_sheet.png"
        sheet.save(sheet_path)
    else:
        sheet_path = args.output_dir / "recovery_shadow_plot_sheet.png"

    report = {
        "trace_output": str(args.trace_output),
        "output_dir": str(args.output_dir),
        "sheet_path": str(sheet_path),
        "top_k": int(args.top_k),
        "trajectories": hard,
    }
    (args.output_dir / "recovery_shadow_plot_report.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
