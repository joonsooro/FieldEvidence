from __future__ import annotations

from pathlib import Path
from typing import Optional
import math
import cv2  # type: ignore


def create_snip(
    src_mp4: Path,
    center_s: float,
    pre_s: float,
    post_s: float,
    out_path: Path,
) -> Optional[Path]:
    # Lazy import to avoid hard dependency at import time
    cap = cv2.VideoCapture(str(src_mp4))
    if not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    dur_s = total / fps if total > 0 else 0.0

    start_s = max(0.0, center_s - pre_s)
    end_s = min(dur_s, center_s + post_s)
    if end_s <= start_s:
        end_s = min(dur_s, start_s + pre_s + post_s)

    start_f = int(math.floor(start_s * fps))
    end_f = int(math.ceil(end_s * fps))
    end_f = min(end_f, total if total > 0 else end_f)

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 360)
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))
    f = start_f
    while f < end_f:
        ok, frame = cap.read()
        if not ok:
            break
        writer.write(frame)
        f += 1
    writer.release()
    cap.release()
    return out_path if out_path.exists() else None
