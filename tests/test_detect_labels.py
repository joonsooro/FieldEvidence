from pathlib import Path
import tempfile

from src.detect.labels import detect_from_csv_labels


def test_detect_from_csv_labels_minimal():
    csv_text = "frame,label,score\n130,person,0.92\n151,none,0.1\n"
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "c.csv"
        p.write_text(csv_text, encoding="utf-8")
        evs = detect_from_csv_labels(p)
        assert any(abs(e["frame_idx"] - 130) <= 20 for e in evs)

