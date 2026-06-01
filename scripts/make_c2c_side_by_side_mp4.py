#!/usr/bin/env python3
"""Create a labeled side-by-side comparison mp4 from two source videos."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

try:  # pragma: no cover
    from moviepy.editor import ImageClip, VideoFileClip, clips_array  # type: ignore
except Exception:  # pragma: no cover
    from moviepy import ImageClip, VideoFileClip, clips_array  # type: ignore
from PIL import Image, ImageDraw, ImageFont


def _load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for candidate in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
    ]:
        path = Path(candidate)
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def _make_label_clip(text: str, width: int, height: int, duration: float) -> ImageClip:
    image = Image.new("RGB", (width, height), (18, 18, 18))
    draw = ImageDraw.Draw(image)
    font = _load_font(max(18, min(28, height - 16)))
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    x = max(16, (width - text_w) // 2)
    y = max(8, (height - text_h) // 2 - 2)
    draw.text((x, y), text, fill=(245, 245, 245), font=font)
    clip = ImageClip(np.asarray(image))
    if hasattr(clip, "with_duration"):
        return clip.with_duration(duration)  # type: ignore[return-value]
    return clip.set_duration(duration)  # type: ignore[return-value]


def _fit_clip(clip: VideoFileClip, height: int) -> VideoFileClip:
    if hasattr(clip, "resized"):
        return clip.resized(height=height)  # type: ignore[return-value]
    return clip.resize(height=height)  # type: ignore[return-value]


def _resize_width(clip: VideoFileClip, width: int) -> VideoFileClip:
    if hasattr(clip, "resized"):
        return clip.resized(width=width)  # type: ignore[return-value]
    return clip.resize(width=width)  # type: ignore[return-value]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left", required=True, help="Left mp4 path")
    parser.add_argument("--right", required=True, help="Right mp4 path")
    parser.add_argument("--left-label", required=True, help="Label for left panel")
    parser.add_argument("--right-label", required=True, help="Label for right panel")
    parser.add_argument("--output", required=True, help="Output mp4 path")
    parser.add_argument("--label-height", type=int, default=54, help="Top label strip height")
    parser.add_argument("--fps", type=int, default=30, help="Output fps")
    args = parser.parse_args()

    left = VideoFileClip(args.left)
    right = VideoFileClip(args.right)
    target_height = min(left.h, right.h)
    left = _fit_clip(left, target_height)
    right = _fit_clip(right, target_height)

    panel_width = max(left.w, right.w)
    left = _resize_width(left, panel_width)
    right = _resize_width(right, panel_width)

    label_height = int(args.label_height)
    duration = min(left.duration, right.duration)
    if hasattr(left, "with_duration"):
        left = left.with_duration(duration)  # type: ignore[assignment]
        right = right.with_duration(duration)  # type: ignore[assignment]
    else:
        left = left.set_duration(duration)  # type: ignore[assignment]
        right = right.set_duration(duration)  # type: ignore[assignment]
    left_label = _make_label_clip(args.left_label, left.w, label_height, duration)
    right_label = _make_label_clip(args.right_label, right.w, label_height, duration)

    final = clips_array([[left_label, right_label], [left, right]])
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    final.write_videofile(
        str(out_path),
        fps=args.fps,
        codec="libx264",
        audio=False,
        preset="medium",
        threads=4,
        ffmpeg_params=["-pix_fmt", "yuv420p"],
    )


if __name__ == "__main__":
    main()
