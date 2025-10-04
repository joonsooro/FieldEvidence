import json
from pathlib import Path

from src.c2.mock_c2 import stream_feed


def test_events_stream_normalizes_and_skips_malformed(tmp_path: Path):
    events_path = tmp_path / "events.jsonl"
    out_path = tmp_path / "out" / "c2_feed.jsonl"

    lines = [
        {
            "uid": 42,
            "ts_ns": 1234567890123,
            "lat": 37.4219999,
            "lon": -122.0840575,
            "event_code": 2,
            "conf": 0.9,
        },
        {
            "uid": 43,
            "ts_ns": 9876543210,
            "lat": 37.0,
            "lon": -122.0,
            "event_code": 3,
            "intent_score": 0.75,
        },
    ]

    events_path.parent.mkdir(parents=True, exist_ok=True)
    with events_path.open("w", encoding="utf-8") as f:
        for d in lines:
            f.write(json.dumps(d) + "\n")
        # malformed line
        f.write("{ this is not json\n")

    # Run the stream function (no follow)
    stream_feed(pings_path=None, events_path=events_path, out_path=out_path, follow=False)

    # Validate output
    assert out_path.exists()
    with out_path.open("r", encoding="utf-8") as f:
        out_lines = [json.loads(x) for x in f.read().splitlines() if x.strip()]

    assert len(out_lines) == 2

    # Check normalized keys present
    for rec in out_lines:
        assert set(rec.keys()) == {
            "event_id",
            "uid",
            "ts_ns",
            "lat",
            "lon",
            "event_code",
            "conf",
            "intent_score",
        }

    # Preserve ordering (first two valid lines)
    assert out_lines[0]["uid"] == 42
    assert out_lines[0]["conf"] == 0.9
    assert out_lines[0]["intent_score"] == 0.0  # defaulted

    assert out_lines[1]["uid"] == 43
    assert out_lines[1]["conf"] == 0.5  # defaulted
    assert out_lines[1]["intent_score"] == 0.75

