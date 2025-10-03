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
from src.infra.snips import create_snip

import xml.etree.ElementTree as ET
from datetime import datetime, timezone
import re


PINGS_PATH = Path("out/pings.bin")
MONITOR_PATH = Path("out/monitor.jsonl")
COT_DIR = Path("out/cot")


# --- Thread state ---
_LOCK = threading.Lock()
_THREAD: Optional[threading.Thread] = None
_COT_THREAD: Optional[threading.Thread] = None
_STOP = threading.Event()
_AUTO_ENABLED = True  # legacy flag used by scheduler; actual toggling handled by _TOGGLE_THREAD
_FLIP_COUNTDOWN = 5

# --- Background comm disruption toggler (independent of replay scheduler) ---
_TOGGLE_THREAD: Optional[threading.Thread] = None
_TOGGLE_STOP = threading.Event()
_AUTO = True


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


def _write_monitor_snapshot_basic(online: bool, lat_e2e: List[float] | None = None, lat_drn: List[float] | None = None) -> None:
    """Write a lightweight monitor snapshot so UI reflects ONLINE/OFFLINE even without replay.
    Latencies may be empty and will be used only if provided.
    """
    try:
        now_ns = time.monotonic_ns()
        d = store_forward.depth()
        oldest_ms: Optional[int] = None
        pk = store_forward.peek_oldest(1)
        if pk:
            _, ts_ns, _ = pk[0]
            oldest_ms = int((now_ns - int(ts_ns)) / 1e6)

        sample_e2e = list(lat_e2e or [])
        sample_drn = list(lat_drn or [])
        p50_e2e = _qtile(sample_e2e, 0.50) if sample_e2e else 0.0
        p95_e2e = _qtile(sample_e2e, 0.95) if sample_e2e else 0.0
        p50_drn = _qtile(sample_drn, 0.50) if sample_drn else 0.0
        p95_drn = _qtile(sample_drn, 0.95) if sample_drn else 0.0

        MONITOR_PATH.parent.mkdir(parents=True, exist_ok=True)
        line = {
            "t_ns": now_ns,
            "depth": d,
            "sent_tot": 0,
            "sent_1s": 0,
            "oldest_ms": oldest_ms,
            "p50_e2e_ms": p50_e2e,
            "p95_e2e_ms": p95_e2e,
            "p50_drain_ms": p50_drn,
            "p95_drain_ms": p95_drn,
            "online": bool(online),
        }
        with MONITOR_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(line) + "\n")
    except Exception:
        # ignore any write errors to keep demo resilient
        pass


def start_toggler() -> None:
    """Start the background connectivity toggler if not already running."""
    global _TOGGLE_THREAD
    if _TOGGLE_THREAD and _TOGGLE_THREAD.is_alive():
        return
    _TOGGLE_STOP.clear()
    th = threading.Thread(target=_run_toggler, daemon=True)
    _TOGGLE_THREAD = th
    th.start()


def stop_toggler() -> None:
    """Stop the background connectivity toggler."""
    _TOGGLE_STOP.set()


def _run_toggler() -> None:
    """Flip OFF → ON every 5s while enabled; write minimal monitor snapshots.
    When online, perform a short non-blocking drain tick so queue can advance.
    """
    state = False  # start OFF then ON, repeat
    while not _TOGGLE_STOP.is_set() and _AUTO:
        try:
            set_online(state)
            lat_e2e: List[float] = []
            lat_drn: List[float] = []
            if state:
                try:
                    snap = store_forward.drain(send_fn=send_http, limit=200, budget_ms=50)
                    lat_e2e = list(snap.get("latency_ms", []))
                    lat_drn = list(snap.get("drain_ms", []))
                except Exception:
                    pass
            # write a lightweight monitor snapshot reflecting current state
            _write_monitor_snapshot_basic(state, lat_e2e, lat_drn)
        except Exception:
            pass
        # sleep 5s in responsive 0.1s increments
        for _ in range(50):
            if _TOGGLE_STOP.is_set() or not _AUTO:
                break
            time.sleep(0.1)
        state = not state


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

        # Initialize connectivity; toggler thread controls ON/OFF. Default to ONLINE.
        set_online(True)

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

            # Disruption handled by independent toggler; keep scheduler agnostic

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

            # no pattern-based countdown; handled above

    finally:
        with _LOCK:
            # Mark thread reference as cleared when exiting
            _THREAD = None


def _parse_event_time_iso8601(xml_path: Path) -> Optional[float]:
    try:
        txt = xml_path.read_text(encoding="utf-8")
        root = ET.fromstring(txt)
        if root.tag != "event":
            return None
        t = root.attrib.get("time", "")
        if not t:
            return None
        # Support trailing 'Z'
        ts = t
        if ts.endswith("Z"):
            ts = ts[:-1] + "+00:00"
        dt = datetime.fromisoformat(ts)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return float(dt.timestamp())
    except Exception:
        return None


def _pick_source_video() -> Optional[Path]:
    viz = Path("out/viz")
    best = viz / "clip_best.mp4"
    try:
        if best.exists():
            return best
        if viz.exists():
            vids = sorted(viz.glob("*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True)
            return vids[0] if vids else None
    except Exception:
        return None
    return None


def _video_props(mp4: Path) -> tuple[float, int, float]:
    """Return (fps, total_frames, duration_s)."""
    import cv2

    cap = cv2.VideoCapture(str(mp4))
    if not cap.isOpened():
        return (30.0, 0, 0.0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    dur = (total / fps) if total > 0 else 0.0
    cap.release()
    return (fps, total, dur)


def _next_seq_number_under(dir_path: Path) -> int:
    dir_path.mkdir(parents=True, exist_ok=True)
    pat = re.compile(r"^clip_(\d{2})")
    mx = -1
    for p in dir_path.glob("*.mp4"):
        m = pat.match(p.name)
        if m:
            try:
                mx = max(mx, int(m.group(1)))
            except Exception:
                pass
    return max(0, mx + 1)


def _make_snip_from_latest(mp4: Path) -> Optional[Path]:
    fps, total, dur_s = _video_props(mp4)
    center_s = max(2.1, dur_s - 0.5)
    seq = _next_seq_number_under(Path("out/snips"))
    frame_tag = int(center_s * fps)
    outp = Path("out/snips") / f"clip_{seq:02d}_f{frame_tag:04d}.mp4"
    try:
        return create_snip(mp4, center_s, 2.0, 2.0, outp)
    except Exception:
        return None


def _cot_watcher() -> None:
    global _COT_THREAD
    seen: set[str] = set()
    try:
        while not _STOP.is_set():
            # scan directory
            try:
                COT_DIR.mkdir(parents=True, exist_ok=True)
                files = sorted(COT_DIR.glob("*.xml"), key=lambda p: p.stat().st_mtime)
            except Exception:
                files = []

            new_files = [p for p in files if str(p) not in seen]
            for p in new_files:
                seen.add(str(p))
                # parse event time for potential future mapping (unused in fallback)
                _ = _parse_event_time_iso8601(p)

                # pick source video and schedule snip 2s later
                src = _pick_source_video()
                if src is None:
                    continue

                def delayed_snip(mp4_src: Path) -> None:
                    _make_snip_from_latest(mp4_src)

                t = threading.Timer(2.0, delayed_snip, args=(src,))
                t.daemon = True
                t.start()

            # poll interval ~1s
            _STOP.wait(0.8)
    finally:
        with _LOCK:
            _COT_THREAD = None


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
        # legacy flag retained for compatibility; independent toggler controls actual comm flips
        _AUTO_ENABLED = True
        # Start scheduler
        th = threading.Thread(target=_scheduler, args=(cfg,), daemon=True)
        _THREAD = th
        th.start()
        # Start CoT watcher
        cot_th = threading.Thread(target=_cot_watcher, daemon=True)
        _COT_THREAD = cot_th
        cot_th.start()


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
    # also stop watcher
    with _LOCK:
        cot_th = _COT_THREAD
    if cot_th is not None:
        cot_th.join(timeout=1.0)
    with _LOCK:
        _COT_THREAD = None


def is_running() -> bool:
    with _LOCK:
        return _THREAD is not None and _THREAD.is_alive()


def set_auto_disruption(flag: bool) -> None:
    """Enable/disable comm disruption.
    ON = flip OFF/ON every 5s via background toggler; OFF = force ONLINE and stop toggler.
    """
    global _AUTO_ENABLED, _FLIP_COUNTDOWN, _AUTO
    _AUTO = bool(flag)
    _AUTO_ENABLED = _AUTO  # keep legacy flag aligned
    if _AUTO:
        start_toggler()
        with _LOCK:
            _FLIP_COUNTDOWN = 5
    else:
        stop_toggler()
        try:
            set_online(True)  # force ONLINE
            _write_monitor_snapshot_basic(True)
        except Exception:
            pass


def get_auto_disruption() -> bool:
    return bool(_AUTO_ENABLED)


# --------- Auto snip creation hook ---------
import os

_SNIPS_DIR = Path("out/snips")
_SNIPS_DIR.mkdir(parents=True, exist_ok=True)


def on_event_emitted(ev: dict) -> None:
    """
    Called when an event is emitted during replay.
    Expected keys (best-effort): 'video_path', 'frame_idx', 'fps', 'clip_id', 'reliability'
    If missing, falls back to latest known video and fps=30.0.
    """
    try:
        # Lazy import to avoid hard dependency at import time
        import cv2  # type: ignore
        video_path = ev.get("video_path") or _guess_latest_video()
        if not video_path or not Path(video_path).exists():
            return

        fps = float(ev.get("fps") or 30.0)
        f_center = int(ev.get("frame_idx") or 0)
        span = int(2 * fps)  # ±2s
        f_start = max(0, f_center - span)
        f_end = f_center + span

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 360)
        if f_end >= total and total > 0:
            f_end = total - 1

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        clip_id = ev.get("clip_id") or "00"
        out_name = f"clip_{clip_id}_f{f_center:05d}.mp4"
        out_path = _SNIPS_DIR / out_name
        vw = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))

        cap.set(cv2.CAP_PROP_POS_FRAMES, f_start)
        for _ in range(max(0, f_end - f_start + 1)):
            ok, frame = cap.read()
            if not ok:
                break
            vw.write(frame)

        vw.release()
        cap.release()
    except Exception:
        # swallow all errors; UI will simply have no new snip
        pass


def _guess_latest_video() -> Optional[str]:
    # Prefer out/viz/*.mp4, fallback to any *.mp4 under out/
    try:
        for folder in ["out/snips", "out/viz", "out"]:
            cand = list(Path(folder).glob("*.mp4"))
            if cand:
                cand = sorted(cand, key=lambda p: p.stat().st_mtime, reverse=True)
                return str(cand[0])
    except Exception:
        return None
    return None
