from __future__ import annotations

from pathlib import Path
import csv


def detect_from_csv_labels(csv_path: Path) -> list[dict]:
    """
    Minimal labels-only detector.
    Accepts CSV with columns: frame,label,score
    Considers label != 'none' and score >= 0.5 as an event.
    Returns list of dicts: {"frame_idx": int}
    """
    events: list[dict] = []
    with csv_path.open("r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                frame = int(row.get("frame", 0))
            except Exception:
                continue
            label = (row.get("label") or "").strip().lower()
            try:
                score = float(row.get("score") or 0.0)
            except Exception:
                score = 0.0
            if label != "none" and score >= 0.5:
                events.append({"frame_idx": frame})
    return events

