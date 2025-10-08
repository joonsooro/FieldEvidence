from __future__ import annotations
import argparse, time, json
from pathlib import Path
import sys
from typing import List, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.infra.store_forward import enqueue, drain, depth, peek_oldest
from src.infra.net_sim import set_online, is_online, send_http
from src.infra import paths

MON_PATH = paths.MONITOR

def read_varint(f):
    shift = 0; out = 0
    while True:
        b = f.read(1)
        if not b: return None
        b = b[0]
        out |= (b & 0x7F) << shift
        if not (b & 0x80): return out
        shift += 7

def load_pings(bin_path: Path) -> List[bytes]:
    out: List[bytes] = []
    with bin_path.open("rb") as f:
        while True:
            n = read_varint(f)
            if n is None: break
            out.append(f.read(n))
    return out

def parse_pattern(s: str) -> List[Tuple[int, bool]]:
    parts = [p.strip() for p in s.split(",") if p.strip()]
    out: List[Tuple[int, bool]] = []
    for p in parts:
        dur, state = p.split(":")
        out.append((int(dur), state.lower() in ("on","1","true")))
    return out

def qtile(vals: List[float], q: float) -> float:
    if not vals: return 0.0
    vals = sorted(vals)
    i = int((len(vals)-1)*q)
    return float(vals[i])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", default=str(paths.PINGS))
    ap.add_argument("--pattern", default="5:off,5:on")
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--budget_ms", type=int, default=50)
    args = ap.parse_args()

    ev_path = Path(args.events)
    if not ev_path.exists():
        print(f"[ERR] events not found: {ev_path}"); return 2

    # Record existing queue depth (items already enqueued)
    existing0 = depth()

    payloads = load_pings(ev_path)
    now_enq_ns = time.monotonic_ns()
    for pb in payloads:
        enqueue(pb, ts_ns=now_enq_ns)
    enq_total = len(payloads)
    print(f"[ENQ] enqueued {enq_total} payload(s).")

    # Total expected to send this session (existing + newly enqueued)
    total_expected = existing0 + enq_total

    pattern = parse_pattern(args.pattern)
    if not pattern:
        print("[ERR] empty pattern"); return 2

    # Initialize schedule
    phase_idx = 0
    phase_remaining = max(0, pattern[0][0])
    set_online(pattern[0][1])
    print(f"[NET] {'ONLINE' if is_online() else 'offline'} for {phase_remaining}s")

    # Preserve previous non-empty samples
    last_latencies: List[float] = []         # E2E latency (last non-empty sample)
    last_drain_latencies: List[float] = []   # drain-only latency (last non-empty sample)
    # Cumulative sends (sum only what this process sent)
    sent_tot_acc = 0

    MON_PATH.parent.mkdir(parents=True, exist_ok=True)

    next_t = time.monotonic()
    while True:
        now = time.monotonic()
        if now < next_t:
            time.sleep(max(0.0, next_t - now))
            now = time.monotonic()
        next_t += 1.0
        now_ns = time.monotonic_ns()

        # Per-tick measurement containers (always reset)
        lat_e2e: List[float] = []
        lat_drn: List[float] = []

        # Phase handling
        if phase_remaining <= 0:
            phase_idx += 1
            if phase_idx >= len(pattern):
                break
            phase_remaining = max(0, pattern[phase_idx][0])
            set_online(pattern[phase_idx][1])
            print(f"[NET] {'ONLINE' if is_online() else 'offline'} for {phase_remaining}s")

        # Non-blocking drain only when online
        sent_this_tick = 0
        if is_online():
            snap = drain(
                send_fn=send_http,
                limit=args.limit,
                budget_ms=args.budget_ms,
                now_ns=now_ns,
                require_online=is_online,
            )
            lat_e2e = list(snap.get("latency_ms", []))
            lat_drn = list(snap.get("drain_ms", []))
            sent_this_tick = int(snap.get("sent", 0))

        # Metrics
        d = depth()
        sent_tot_acc = max(0, sent_tot_acc + sent_this_tick)
        sent_1s = sent_this_tick

        # Oldest pending age (monotonic)
        oldest_ms = None
        pk = peek_oldest(1)
        if pk:
            _, ts_ns, _ = pk[0]
            oldest_ms = int((now_ns - int(ts_ns)) / 1e6)

        # Per-tick latency quantiles (E2E & drain-only) with last non-empty fallback
        sample_e2e = lat_e2e if lat_e2e else last_latencies
        sample_drn = lat_drn if lat_drn else last_drain_latencies
        p50_e2e = qtile(sample_e2e, 0.50) if sample_e2e else 0.0
        p95_e2e = qtile(sample_e2e, 0.95) if sample_e2e else 0.0
        p50_drn = qtile(sample_drn, 0.50) if sample_drn else 0.0
        p95_drn = qtile(sample_drn, 0.95) if sample_drn else 0.0

        if lat_e2e:
            last_latencies = lat_e2e
        if lat_drn:
            last_drain_latencies = lat_drn

        # Log and monitor snapshot
        print(
            f"[TICK] depth={d:4d} sent_tot={sent_tot_acc:4d} sent_1s={sent_1s:3d} "
            f"oldest_ms={oldest_ms} "
            f"p50_e2e={p50_e2e:.1f}ms p95_e2e={p95_e2e:.1f}ms "
            f"p50_drn={p50_drn:.2f}ms p95_drn={p95_drn:.2f}ms "
            f"online={is_online()}"
        )

        snap_line = {
            "t_ns": now_ns,
            "depth": d,
            "sent_tot": sent_tot_acc,
            "sent_1s": sent_1s,
            "oldest_ms": oldest_ms,
            "p50_e2e_ms": p50_e2e,
            "p95_e2e_ms": p95_e2e,
            "p50_drain_ms": p50_drn,
            "p95_drain_ms": p95_drn,
            "online": is_online(),
        }
        with MON_PATH.open("a") as f:
            f.write(json.dumps(snap_line) + "\n")

        phase_remaining -= 1

    print("[DONE] replay finished."); return 0

if __name__ == "__main__":
    raise SystemExit(main())
