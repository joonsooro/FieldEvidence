from __future__ import annotations

import json
import threading
from threading import Thread
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Tuple

from src.infra import store_forward
from src.infra.net_sim import is_online, send_http, set_online
from src.infra.cot_encode import microping_to_cot_xml, write_cot_file
from src.infra import paths
from src.wire import salute_pb2
from src.wire.codec import read_ld_stream
from src.infra.snips import create_snip

import xml.etree.ElementTree as ET
from datetime import datetime, timezone
import re


PINGS_PATH = paths.PINGS
MONITOR_PATH = paths.MONITOR
COT_DIR = paths.COT_DIR

# Last time a monitor line was written (monotonic ns)
_LAST_WRITE_NS: int = 0

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

# --- Heartbeat thread (1 Hz monitor snapshots) ---
_HB_THREAD: Optional[threading.Thread] = None
_HB_STOP = threading.Event()

# Per-process counters for heartbeat
_SENT_TOT: int = 0
_SENT_1S: int = 0


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
        _mark_write()
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
    """Flip OFF → ON every 5s while enabled, writing a flip snapshot each time.
    Do not drain here; scheduler handles draining and metrics.
    """
    state = False  # start OFF then ON, repeat
    while not _TOGGLE_STOP.is_set() and _AUTO:
        try:
            set_online(state)
        except Exception:
            pass
        # Sleep ~5s in responsive 0.1s increments
        for _ in range(50):
            if _TOGGLE_STOP.is_set() or not _AUTO:
                break
            time.sleep(0.1)
        state = not state


def _run_heartbeat() -> None:
    """Write a monitor snapshot every second, regardless of online state.
    Uses per-process counters updated by the scheduler.
    """
    global _SENT_1S
    next_t = time.monotonic()
    while not _HB_STOP.is_set():
        now = time.monotonic()
        if now < next_t:
            _HB_STOP.wait(max(0.0, next_t - now))
            now = time.monotonic()
        next_t += 1.0
        now_ns = time.monotonic_ns()
        try:
            d = store_forward.depth()
            oldest_ms: Optional[int] = None
            pk = store_forward.peek_oldest(1)
            if pk:
                _, ts_ns, _ = pk[0]
                oldest_ms = int((now_ns - int(ts_ns)) / 1e6)

            # For simplicity, latency quantiles are 0 unless separately tracked
            line = {
                "t_ns": now_ns,
                "depth": d,
                "sent_tot": max(0, int(_SENT_TOT)),
                "sent_1s": max(0, int(_SENT_1S)),
                "oldest_ms": oldest_ms,
                "p50_e2e_ms": 0.0,
                "p95_e2e_ms": 0.0,
                "p50_drain_ms": 0.0,
                "p95_drain_ms": 0.0,
                "online": is_online(),
            }
            MONITOR_PATH.parent.mkdir(parents=True, exist_ok=True)
            with MONITOR_PATH.open("a", encoding="utf-8") as f:
                f.write(json.dumps(line) + "\n")
            _mark_write()
        except Exception:
            pass
        finally:
            # reset per-second counter
            _SENT_1S = 0


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
        for pb in payloads:
            store_forward.enqueue(pb, ts_ns=enq_t_ns)

        # Ensure output dirs
        MONITOR_PATH.parent.mkdir(parents=True, exist_ok=True)
        COT_DIR.mkdir(parents=True, exist_ok=True)

        # Rolling samples and counters for accurate metrics
        last_e2e: List[float] = []
        last_drn: List[float] = []
        sent_tot_acc = 0

        next_t = time.monotonic()
        while not _STOP.is_set():
            now = time.monotonic()
            if now < next_t:
                time.sleep(max(0.0, next_t - now))
                now = time.monotonic()
            next_t += 1.0
            now_ns = time.monotonic_ns()

            # Drain only when online; collect latencies for this tick
            lat_e2e: List[float] = []
            lat_drn: List[float] = []
            sent_this_tick = 0
            if is_online():
                try:
                    snap = store_forward.drain(
                        send_fn=_send_and_write_cot,
                        limit=cfg.limit,
                        budget_ms=cfg.budget_ms,
                        now_ns=now_ns,
                        require_online=is_online,
                    )
                    lat_e2e = list(snap.get("latency_ms", []))
                    lat_drn = list(snap.get("drain_ms", []))
                    sent_this_tick = int(snap.get("sent", 0))
                except Exception:
                    sent_this_tick = 0
                    lat_e2e = []
                    lat_drn = []
            else:
                sent_this_tick = 0
                lat_e2e = []
                lat_drn = []

            # Accumulate counters safely
            sent_tot_acc = max(0, sent_tot_acc + sent_this_tick)
            sent_1s = sent_this_tick

            # Oldest pending age
            oldest_ms: Optional[int] = None
            pk = store_forward.peek_oldest(1)
            if pk:
                _, ts_ns, _ = pk[0]
                oldest_ms = int((now_ns - int(ts_ns)) / 1e6)

            # Quantiles (fallback to last non-empty)
            sample_e2e = lat_e2e if lat_e2e else last_e2e
            sample_drn = lat_drn if lat_drn else last_drn
            p50_e2e = _qtile(sample_e2e, 0.50)
            p95_e2e = _qtile(sample_e2e, 0.95)
            p50_drn = _qtile(sample_drn, 0.50)
            p95_drn = _qtile(sample_drn, 0.95)

            # Update last samples if any were recorded this tick
            if lat_e2e:
                last_e2e = lat_e2e
            if lat_drn:
                last_drn = lat_drn

            # Write 1 Hz monitor snapshot
            line = {
                "t_ns": now_ns,
                "depth": store_forward.depth(),
                "sent_tot": sent_tot_acc,
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
                _mark_write()
            except Exception:
                pass

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
    """Find the most recent mp4 in viz/, fallback to snips/."""
    for folder in [paths.VIZ_DIR, paths.SNIPS_DIR]:
        try:
            if folder.exists():
                vids = sorted(folder.glob("*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True)
                if vids:
                    return vids[0]
        except Exception:
            continue
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


def _make_snip_from_event(src: Path, event_ts_s: float, margin_s: float = 2.0) -> Optional[Path]:
    """Create a ±margin_s snip around event_ts_s (in seconds) from src video."""
    import cv2
    fps, total, dur_s = _video_props(src)
    if fps <= 0.0 or total <= 0 or dur_s <= 0.0:
        return None

    # Clamp center into [0, dur_s]
    center_s = max(0.0, min(float(event_ts_s), dur_s))
    f_center = int(round(center_s * fps))
    span = int(round(margin_s * fps))
    f_start = max(0, f_center - span)
    f_end = min(total - 1, f_center + span)

    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        return None

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 360)
    # Prefer H.264/avc1 for browser compatibility; fallback to mp4v if unavailable
    fourcc_try = ["avc1", "mp4v"]

    seq = _next_seq_number_under(paths.SNIPS_DIR)
    outp = paths.SNIPS_DIR / f"clip_{seq:02d}_f{f_center:05d}.mp4"
    vw = None
    for cc in fourcc_try:
        fourcc = cv2.VideoWriter_fourcc(*cc)
        vw = cv2.VideoWriter(str(outp), fourcc, fps, (width, height))
        if vw is not None and vw.isOpened():
            break
        try:
            vw.release()
        except Exception:
            pass
        vw = None
    if vw is None or not vw.isOpened():
        cap.release()
        return None

    # Seek to start frame and write through end frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, f_start)
    frames_to_write = max(0, f_end - f_start + 1)
    written = 0
    while written < frames_to_write:
        ok, frame = cap.read()
        if not ok:
            break
        vw.write(frame)
        written += 1

    vw.release()
    cap.release()
    # Validate non-empty file
    try:
        if outp.exists() and outp.stat().st_size > 0:
            return outp
    except Exception:
        pass
    return None
    
def delayed_snip(mp4_src: Path) -> None:
    _make_snip_from_event(mp4_src)


def _cot_watcher() -> None:
    """
    Watch new CoT files, parse event time, and call on_event_emitted().
    """
    global _COT_THREAD
    seen: set[str] = set()
    try:
        while not _STOP.is_set():
            try:
                COT_DIR.mkdir(parents=True, exist_ok=True)
                files = sorted(COT_DIR.glob("*.xml"), key=lambda p: p.stat().st_mtime)
            except Exception:
                files = []

            new_files = [p for p in files if str(p) not in seen]
            for p in new_files:
                seen.add(str(p))
                ev_ts = _parse_event_time_iso8601(p)
                if ev_ts is None:
                    continue
                src = _pick_source_video()
                ev = {
                    "ts_ns": int(ev_ts * 1e9),
                    "video_path": str(src) if src else None,
                }
                on_event_emitted(ev)

            _STOP.wait(0.8)
    finally:
        with _LOCK:
            _COT_THREAD = None


def start_replay(pattern: str = "5:off,5:on", limit: int = 200, budget_ms: int = 50) -> None:
    global _THREAD, _STOP, _AUTO_ENABLED
    with _LOCK:
        if _THREAD is not None and _THREAD.is_alive():
            return
        # --- clear old monitor lines for a clean demo run ---
        try:
            MONITOR_PATH.parent.mkdir(parents=True, exist_ok=True)
            MONITOR_PATH.write_text("", encoding="utf-8")
        except Exception:
            pass
        # ----------------------------------------------------
        cfg = _Config(pattern=_parse_pattern(pattern), limit=int(limit), budget_ms=int(budget_ms))
        _STOP.clear()
        _AUTO_ENABLED = True
        th = threading.Thread(target=_scheduler, args=(cfg,), daemon=True)
        _THREAD = th
        th.start()
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
    # no separate heartbeat to stop


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


def ensure_replay_running(limit: int = 200, budget_ms: int = 50) -> None:
    """Start the scheduler if not alive, with standard defaults."""
    if not is_running():
        start_replay(pattern="5:off,5:on", limit=limit, budget_ms=budget_ms)
    else:
        pass


def _toggler_alive() -> bool:
    th = _TOGGLE_THREAD
    return bool(th and th.is_alive())


def ensure_toggler_running(auto: bool) -> None:
    """Ensure toggler state matches `auto` and write a snapshot when disabling."""
    global _AUTO
    _AUTO = bool(auto)
    if _AUTO:
        if not _toggler_alive():
            start_toggler()
    else:
        stop_toggler()
        try:
            set_online(True)
            _write_monitor_snapshot_basic(True)
        except Exception:
            pass


def _mark_write() -> None:
    global _LAST_WRITE_NS
    try:
        _LAST_WRITE_NS = time.monotonic_ns()
    except Exception:
        pass


def start_heartbeat() -> None:  # legacy no-op
    return None


def stop_heartbeat() -> None:  # legacy no-op
    return None


# --------- Auto snip creation hook ---------
import os

_SNIPS_DIR = paths.SNIPS_DIR
_SNIPS_DIR.mkdir(parents=True, exist_ok=True)


def on_event_emitted(ev: dict) -> None:
    """
    Called when an event is emitted. Creates a ±2s snip into out/snips/ and returns immediately.
    Expected in ev:
      - 'ts_ns' (preferred) or 'time' in seconds
      - optional 'video_path'
    """
    from threading import Thread
    try:
        # derive event time in seconds
        if "ts_ns" in ev:
            event_ts_s = float(ev["ts_ns"]) / 1e9
        elif "time" in ev:
            event_ts_s = float(ev["time"])
        else:
            return

        # pick source video (prefer viz/, fallback snips/)
        video_path = ev.get("video_path")
        if not video_path:
            src = _pick_source_video()
        else:
            src = Path(video_path) if Path(video_path).exists() else _pick_source_video()
        if not src:
            return

        # spawn background worker (non-blocking)
        Thread(target=_make_snip_from_event, args=(src, event_ts_s, 2.0), daemon=True).start()
    except Exception:
        # swallow to avoid breaking the replay loop
        pass

def _guess_latest_video() -> Optional[str]:
    # Prefer out/viz/*.mp4, fallback to any *.mp4 under out/
    try:
        for folder in [str(paths.SNIPS_DIR), str(paths.VIZ_DIR), str(paths.OUT)]:
            cand = list(Path(folder).glob("*.mp4"))
            if cand:
                cand = sorted(cand, key=lambda p: p.stat().st_mtime, reverse=True)
                return str(cand[0])
    except Exception:
        return None
    return None
