#!/usr/bin/env python3
from __future__ import annotations
import json, time
from pathlib import Path

OUT = Path("out")

def make_broken_ping(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        f.write(bytes([0xC8, 0x01]))  # malformed varint

def ensure_min_events(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists() or path.stat().st_size == 0:
        path.write_text(json.dumps({
            "ts_ns": 1700000000000000000,
            "lat": 52.5, "lon": 13.3,
            "intent_score": 0.7,
            "uid": "fallback", "event_code": 1
        }) + "\n", encoding="utf-8")

def append_monitor_snapshots():
    m = OUT / "monitor.jsonl"
    m.parent.mkdir(parents=True, exist_ok=True)
    def append(d):
        if m.exists():
            m.write_text(m.read_text(encoding="utf-8") + json.dumps(d) + "\n", encoding="utf-8")
        else:
            m.write_text(json.dumps(d) + "\n", encoding="utf-8")
    t = lambda: int(time.time_ns())
    append({"t_ns": t(), "online": True,  "depth": 2,  "sent_1s": 5, "p95_drain_ms": 200})
    time.sleep(0.05)
    append({"t_ns": t(), "online": False, "depth": 20, "sent_1s": 0, "p95_drain_ms": 2300})
    time.sleep(0.05)
    append({"t_ns": t(), "online": True,  "depth": 3,  "sent_1s": 4, "p95_drain_ms": 1800})

def toggle_label_mode(on=True):
    flag = OUT / "label_mode.on"
    if on: flag.touch()
    else:
        try: flag.unlink()
        except FileNotFoundError: pass

if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    make_broken_ping(OUT / "broken_ping.bin")
    ensure_min_events(OUT / "events.jsonl")
    append_monitor_snapshots()
    toggle_label_mode(True)
    print("drills bootstrap complete")

