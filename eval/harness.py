from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time

from src.infra import store_forward
from src.infra.net_sim import set_online
from src.wire import salute_pb2

try:
    # Prefer local labels utility if present
    from src.detect.labels import detect_from_csv_labels  # type: ignore
except Exception:  # pragma: no cover - fallback stub (should not trigger in tests)
    def detect_from_csv_labels(csv_path: Path) -> list[dict]:
        import csv
        events: list[dict] = []
        with csv_path.open("r", encoding="utf-8") as f:
            r = csv.DictReader(f)
            for row in r:
                try:
                    frame = int(row.get("frame", 0))
                    label = (row.get("label") or "").strip().lower()
                    score = float(row.get("score") or 0.0)
                except Exception:
                    continue
                if label != "none" and score >= 0.5:
                    events.append({"frame_idx": frame})
        return events


REPORT = Path("eval/REPORT.md")


@dataclass
class ClipCase:
    clip_id: str
    csv_path: Path
    incident_frame: int


def run_label_eval(cases: list[ClipCase]) -> dict:
    results = []
    for c in cases:
        evs = detect_from_csv_labels(c.csv_path)
        hit = any(abs(e["frame_idx"] - c.incident_frame) <= 20 for e in evs)
        results.append({
            "clip": c.clip_id,
            "incident": c.incident_frame,
            "hit": hit,
            "n_ev": len(evs),
        })
    pd = sum(1 for r in results if r["hit"]) / max(1, len(results))
    far_proxy = sum(max(0, r["n_ev"] - 1) for r in results) / max(1, len(results))
    return {"results": results, "pd": pd, "far_proxy": far_proxy}


def assert_microping_size_le80() -> dict:
    m = salute_pb2.SalutePing()
    m.uid = 123
    m.ts_ns = 1_700_000_000_000_000_000
    m.lat_q = int(37.422 * 1e7)
    m.lon_q = int(-122.084 * 1e7)
    m.event_code = 1
    payload = m.SerializeToString()
    size = len(payload)
    assert size <= 80, f"microping size {size} > 80"
    return {"microping_size": size}


def drain_round_trip(n: int = 14, max_wait_s: float = 2.0) -> dict:
    store_forward.clear()
    set_online(False)

    # enqueue N
    for i in range(n):
        m = salute_pb2.SalutePing()
        m.uid = i + 1
        m.ts_ns = time.time_ns()
        m.lat_q = 374220000
        m.lon_q = -1220840000
        m.event_code = 1
        store_forward.enqueue(m.SerializeToString(), ts_ns=m.ts_ns)

    depth_off = store_forward.depth()
    assert depth_off == n, f"expected depth {n}, got {depth_off}"

    # go online and drain
    set_online(True)
    t0 = time.monotonic()
    drained = 0
    while time.monotonic() - t0 < max_wait_s:
        snap = store_forward.drain(send_fn=lambda b: None, limit=200, budget_ms=50)
        drained += int(snap.get("sent", 0))
        if store_forward.depth() == 0:
            break
        time.sleep(0.05)

    depth_on = store_forward.depth()
    assert depth_on == 0, f"depth not zero after online: {depth_on}"
    return {
        "enqueued": n,
        "drained": drained,
        "drain_time_ms": int((time.monotonic() - t0) * 1000),
    }


def main() -> None:
    # 1) label eval (provide 5 tiny CSVs below)
    cases = [
        ClipCase("c01", Path("data/labels/c01.csv"), 133),
        ClipCase("c02", Path("data/labels/c02.csv"), 121),
        ClipCase("c03", Path("data/labels/c03.csv"), 148),
        ClipCase("c04", Path("data/labels/c04.csv"), 95),
        ClipCase("c05", Path("data/labels/c05.csv"), 210),
    ]
    label_metrics = run_label_eval(cases)

    # 2) microping size
    size_metrics = assert_microping_size_le80()

    # 3) drain check
    drain_metrics = drain_round_trip(n=14, max_wait_s=2.0)

    # 4) write report
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(
        f"""# Eval Report

## Label Mode (no vision)
- Pd (hit within ±20 frames): **{label_metrics['pd']:.2f}**
- FAR (synthetic proxy): **{label_metrics['far_proxy']:.2f} events/clip**
- Per clip:
{chr(10).join(f"- {r['clip']}: hit={r['hit']} events={r['n_ev']} incident={r['incident']}" for r in label_metrics['results'])}

## Microping Size
- serialized size: **{size_metrics['microping_size']} bytes** (≤ 80 ✅)

## Store & Forward
- enqueued: {drain_metrics['enqueued']}
- drained: {drain_metrics['drained']}
- drain time: ~{drain_metrics['drain_time_ms']} ms (≤ 2000 ms expected)
""",
        encoding="utf-8",
    )
    print("OK: report @", REPORT)


if __name__ == "__main__":
    main()

