from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Tuple

from src.infra import store_forward
from src.infra.net_sim import is_online, send_http, set_online
from src.infra.cot_encode import microping_to_cot_xml, write_cot_file
from src.wire import salute_pb2
from src.wire.codec import read_ld_stream


PINGS_PATH = Path("out/pings.bin")
MONITOR_PATH = Path("out/monitor.jsonl")
COT_DIR = Path("out/cot")


# --- Thread state ---
_LOCK = threading.Lock()
_THREAD: Optional[threading.Thread] = None
_STOP = threading.Event()
_AUTO_ENABLED = True  # whether to auto-toggle connectivity by pattern


@dataclass
class _Config:
    pattern: List[Tuple[int, bool]]  # (seconds, online)
    limit: int
    budget_ms: int


def _parse_pattern(s: str) -> List[Tuple[int, bool]]:
    out: List[Tuple[int, bool]] = []
    if not s:
        return out
    parts = [p.strip() for p in s.split(",") if p.strip()]
    for p in parts:
        try:
            dur_s, state = p.split(":", 1)
            dur = int(dur_s)
            st = state.strip().lower()
            online = st in ("on", "1", "true", "yes")
            out.append((max(0, dur), online))
        except Exception:
            # ignore malformed segments
            continue
    return out


def _qtile(vals: List[float], q: float) -> float:
    if not vals:
        return 0.0
    vs = sorted(vals)
    if q <= 0:
        return float(vs[0])
    if q >= 1:
        return float(vs[-1])
    i = int(round((len(vs) - 1) * q))
    return float(vs[i])


def _load_all_pings(path: Path) -> List[bytes]:
    if not path.exists():
        return []
    out: List[bytes] = []
    with path.open("rb") as fp:
        for blob in read_ld_stream(fp):
            out.append(blob)
    return out


def _send_and_write_cot(payload: bytes) -> None:
    """Wrap network send to also emit a CoT XML file for the payload."""
    # Attempt to send (may raise if offline)
    send_http(payload)

    # On success, parse the microping and write a CoT XML
    msg = salute_pb2.SalutePing()
    msg.ParseFromString(payload)
    xml = microping_to_cot_xml(msg, callsign="UAV-1")
    write_cot_file(xml, COT_DIR, uid=int(msg.uid), ts_ns=int(msg.ts_ns))


def _scheduler(cfg: _Config) -> None:
    global _THREAD
    try:
        # Step 1: enqueue all pings with a common enqueue timestamp
        payloads = _load_all_pings(PINGS_PATH)
        enq_t_ns = time.monotonic_ns()
        existing0 = store_forward.depth()
        for pb in payloads:
            store_forward.enqueue(pb, ts_ns=enq_t_ns)
        enq_total = len(payloads)
        total_expected = existing0 + enq_total
        if total_expected < 0:
            total_expected = 0

        # Init phase schedule
        phase_idx = 0
        phase_remaining = cfg.pattern[0][0] if cfg.pattern else 0
        if cfg.pattern:
            # only set online state on start if auto is enabled
            if _AUTO_ENABLED:
                set_online(cfg.pattern[0][1])

        # Rolling measurements
        last_e2e: List[float] = []
        last_drn: List[float] = []
        sent_prev = total_expected - store_forward.depth()
        if sent_prev < 0:
            sent_prev = 0

        # Ensure output dirs
        MONITOR_PATH.parent.mkdir(parents=True, exist_ok=True)
        COT_DIR.mkdir(parents=True, exist_ok=True)

        next_t = time.monotonic()
        while not _STOP.is_set():
            now = time.monotonic()
            if now < next_t:
                time.sleep(max(0.0, next_t - now))
                now = time.monotonic()
            next_t += 1.0
            now_ns = time.monotonic_ns()

            # Phase / auto-disruption handling
            if _AUTO_ENABLED and cfg.pattern:
                if phase_remaining <= 0:
                    phase_idx += 1
                    if phase_idx >= len(cfg.pattern):
                        phase_idx = 0  # loop pattern for continuous demo
                    phase_remaining = max(0, cfg.pattern[phase_idx][0])
                    set_online(cfg.pattern[phase_idx][1])
                # else keep current state
            # If auto disabled → never flip; keep whatever the current state is

            # Drain non-blocking
            lat_e2e: List[float] = []
            lat_drn: List[float] = []
            sent_this_tick = 0
            try:
                snap = store_forward.drain(
                    send_fn=_send_and_write_cot,
                    limit=cfg.limit,
                    budget_ms=cfg.budget_ms,
                    now_ns=now_ns,
                )
                lat_e2e = list(snap.get("latency_ms", []))
                lat_drn = list(snap.get("drain_ms", []))
                sent_this_tick = int(snap.get("sent", 0))
            except Exception:
                # On any unexpected error, proceed and keep thread alive
                pass

            if lat_e2e:
                last_e2e = lat_e2e
            if lat_drn:
                last_drn = lat_drn

            # Metrics
            d = store_forward.depth()
            sent_tot = total_expected - d
            if sent_tot < 0:
                sent_tot = 0
            sent_1s = sent_tot - sent_prev
            sent_prev = sent_tot

            oldest_ms: Optional[int] = None
            pk = store_forward.peek_oldest(1)
            if pk:
                _, ts_ns, _ = pk[0]
                oldest_ms = int((now_ns - int(ts_ns)) / 1e6)

            sample_e2e = lat_e2e if lat_e2e else last_e2e
            sample_drn = lat_drn if lat_drn else last_drn
            p50_e2e = _qtile(sample_e2e, 0.50)
            p95_e2e = _qtile(sample_e2e, 0.95)
            p50_drn = _qtile(sample_drn, 0.50)
            p95_drn = _qtile(sample_drn, 0.95)

            line = {
                "t_ns": now_ns,
                "depth": d,
                "sent_tot": sent_tot,
                "sent_1s": sent_1s,
                "oldest_ms": oldest_ms,
                "p50_e2e_ms": p50_e2e,
                "p95_e2e_ms": p95_e2e,
                "p50_drain_ms": p50_drn,
                "p95_drain_ms": p95_drn,
                "online": is_online(),
            }
            try:
                with MONITOR_PATH.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(line) + "\n")
            except Exception:
                # ignore write errors to keep demo running
                pass

            if _AUTO_ENABLED and cfg.pattern:
                phase_remaining -= 1

    finally:
        with _LOCK:
            # Mark thread reference as cleared when exiting
            _THREAD = None


def start_replay(pattern: str = "5:off,5:on", limit: int = 200, budget_ms: int = 50) -> None:
    """
    Start the background replay thread if not already running.

    - Reads varint-framed pings from out/pings.bin and enqueues them
    - Every second, toggles network state per pattern when auto-disruption is enabled
    - Drains the queue non-blocking and writes monitor snapshots to out/monitor.jsonl
    - Writes a CoT XML file to out/cot/ for each payload successfully sent
    """
    global _THREAD, _STOP, _AUTO_ENABLED
    with _LOCK:
        if _THREAD is not None and _THREAD.is_alive():
            return
        cfg = _Config(pattern=_parse_pattern(pattern), limit=int(limit), budget_ms=int(budget_ms))
        _STOP.clear()
        # default auto-enabled when starting; UI can adjust via set_auto_disruption()
        _AUTO_ENABLED = True
        th = threading.Thread(target=_scheduler, args=(cfg,), daemon=True)
        _THREAD = th
        th.start()


def stop_replay() -> None:
    """Signal the background replay thread to stop and wait for it to exit."""
    global _THREAD
    with _LOCK:
        th = _THREAD
    if th is None:
        return
    _STOP.set()
    th.join(timeout=2.0)
    with _LOCK:
        _THREAD = None


def is_running() -> bool:
    with _LOCK:
        return _THREAD is not None and _THREAD.is_alive()


def set_auto_disruption(flag: bool) -> None:
    """Enable/disable auto ON/OFF flipping without restarting the thread."""
    global _AUTO_ENABLED
    _AUTO_ENABLED = bool(flag)


def get_auto_disruption() -> bool:
    return bool(_AUTO_ENABLED)

