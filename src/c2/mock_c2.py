from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, Generator, Iterable, Iterator, Optional

from src.wire import salute_pb2
from src.wire.codec import read_ld_stream, decode_ping
import hashlib

# -----------------------------------------------------------------------------
# mock_c2: Normalize inputs (pings.bin/events.jsonl) → unified JSONL feed.
# Goals handled here:
#  - UID type normalization (int/str → output time only, string)
#  - Stable event_id: "{uid_str}#{frame or ts_ms}"
#  - Robust dedup across sources using (uid_str, ts_ns)
#  - Backlog-then-tail for both inputs; truncate/once/follow modes
#  - Never crash on malformed input; warn once per error kind
# -----------------------------------------------------------------------------

LOG = logging.getLogger("mock_c2")

_WARNED_TEXTS: set[str] = set()


# ------------------------------- utils ---------------------------------------
def _uid_to_int(uid_val) -> int:
    # 이미 int면 그대로
    if isinstance(uid_val, int):
        return uid_val
    # 문자열/기타 → sha1 상위 8바이트를 부호없는 64비트 정수로
    b = str(uid_val).encode("utf-8")
    h = hashlib.sha1(b).digest()[:8]
    return int.from_bytes(h, "big", signed=False)

def _suffix_ms_from_ts(ts_ns: int) -> int:
    # ns → ms 반올림
    return int(round(ts_ns / 1_000_000.0))

def _warn_once(msg: str) -> None:
    if msg not in _WARNED_TEXTS:
        _WARNED_TEXTS.add(msg)
        LOG.warning(msg)


def _write_ping_sizes(size: int, path: str = "out/ping_sizes.json", maxlen: int = 500) -> None:
    """Append a size sample and persist rolling p95/max, degrading gracefully."""
    try:
        import json as _json
        from pathlib import Path as _P
        p = _P(path)
        sizes: list[int] = []
        if p.exists():
            try:
                data = _json.loads(p.read_text(encoding="utf-8"))
                sizes = list(data.get("sizes") or [])
            except Exception:
                sizes = []
        sizes.append(int(size))
        sizes = sizes[-int(maxlen):]
        arr = sorted(sizes)
        p95 = arr[int(0.95 * (len(arr) - 1))] if arr else None
        mx = max(arr) if arr else None
        p.write_text(_json.dumps({"sizes": sizes, "p95": p95, "max": mx}, indent=2), encoding="utf-8")
    except Exception:
        # Never fail the pipeline on metrics I/O issues
        pass


def _uid_to_str(uid: Any) -> str:
    """Normalize UID to string for output/dedup without breaking upstream math."""
    try:
        if isinstance(uid, (int,)):
            return str(int(uid))
        if isinstance(uid, float) and uid.is_integer():
            return str(int(uid))
        if isinstance(uid, (bytes, bytearray)):
            try:
                return uid.decode("utf-8", "ignore")
            except Exception:
                return str(uid)
        return str(uid)
    except Exception:
        return str(uid)


def _event_id(uid_str: str, ts_ns: int, frame: Optional[int]) -> str:
    # If frame available (events.jsonl), prefer it; else bucket by ms from ts_ns
    if frame is not None:
        return f"{uid_str}#{int(frame)}"
    ts_ms = int(ts_ns // 1_000_000)
    return f"{uid_str}#{ts_ms}"


def _norm_event(
    *,
    uid: Any,
    ts_ns: int,
    lat: float,
    lon: float,
    event_code: int,
    conf: float = 0.0,
    frame: Optional[int] = None,
    intent_score: float = 0.0,
) -> Dict[str, Any]:
    uid_s = _uid_to_str(uid)
    ev = {
        "event_id": _event_id(uid_s, int(ts_ns), frame),
        "uid": uid_s,
        "ts_ns": int(ts_ns),
        "lat": float(lat),
        "lon": float(lon),
        "event_code": int(event_code),
        "conf": float(conf),
        "intent_score": float(intent_score),
    }
    return ev




# ------------------------------ decoders -------------------------------------

# events 정규화 (normalize_event_dict)
def normalize_event_dict(raw: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    try:
        if "uid" not in raw:
            raise ValueError("missing uid")
        uid_int = _uid_to_int(raw["uid"])

        ts_ns = int(raw["ts_ns"])
        lat = float(raw["lat"])
        lon = float(raw["lon"])
        event_code = int(raw["event_code"])
        conf = float(raw.get("conf", 0.5))
        intent_score = float(raw.get("intent_score", 0.0))

        suffix_ms = _suffix_ms_from_ts(ts_ns)

        return {
            "event_id": f"{uid_int}#{suffix_ms}",
            "uid": uid_int,                   # ← 항상 정수
            "ts_ns": ts_ns,
            "lat": lat,
            "lon": lon,
            "event_code": event_code,
            "conf": conf,
            "intent_score": intent_score,
        }
    except Exception as e:
        _warn_once(f"Skipping malformed event line: {e}")
        return None
# --------------------------- backlog + tailers -------------------------------

def _decode_varint_from_buffer(buf: bytearray, start: int = 0) -> tuple[Optional[int], int]:
    """Decode a varint from buf[start:]. Returns (value, bytes_used). None if incomplete."""
    result = 0
    shift = 0
    idx = start
    while idx < len(buf):
        b = buf[idx]
        result |= (b & 0x7F) << shift
        idx += 1
        if not (b & 0x80):
            return result, idx - start
        shift += 7
        if shift >= 64:
            raise ValueError("Varint too long")
    return None, 0


def iter_jsonl_backlog_then_tail(path: str | Path, follow: bool, once: bool) -> Iterator[Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return iter(())

    def _gen() -> Generator[Dict[str, Any], None, None]:
        with p.open("r", encoding="utf-8") as fp:
            # Backlog
            while True:
                pos = fp.tell()
                line = fp.readline()
                if not line:
                    break
                if not line.endswith("\n"):
                    fp.seek(pos)
                    break
                try:
                    obj = json.loads(line)
                    if isinstance(obj, dict):
                        yield obj
                    else:
                        _warn_once("Skipping malformed JSON line: not an object")
                except json.JSONDecodeError as e:
                    _warn_once(f"Skipping malformed JSON line: {e}")
            if once or not follow:
                return
            # Tail
            while True:
                pos = fp.tell()
                line = fp.readline()
                if not line:
                    time.sleep(0.2)
                    fp.seek(pos)
                    continue
                if not line.endswith("\n"):
                    fp.seek(pos)
                    time.sleep(0.2)
                    continue
                try:
                    obj = json.loads(line)
                    if isinstance(obj, dict):
                        yield obj
                    else:
                        _warn_once("Skipping malformed JSON line: not an object")
                except json.JSONDecodeError as e:
                    _warn_once(f"Skipping malformed JSON line: {e}")

    return _gen()


def iter_pings_backlog_then_tail(path: str | Path, follow: bool, once: bool) -> Iterator[salute_pb2.SalutePing]:
    p = Path(path)
    if not p.exists():
        return iter(())

    def _gen() -> Generator[salute_pb2.SalutePing, None, None]:
        # Backlog using read_ld_stream
        tail_offset = 0
        try:
            with p.open("rb") as f:
                while True:
                    try:
                        for blob in read_ld_stream(f):
                            tail_offset = f.tell()
                            msg = salute_pb2.SalutePing()
                            msg.ParseFromString(blob)
                            yield msg
                        break
                    except ValueError:
                        try:
                            f.seek(tail_offset)
                        except Exception:
                            pass
                        break
        except FileNotFoundError:
            return

        if once or not follow:
            return

        # Tail with incremental varint buffer
        buffer = bytearray()
        while True:
            try:
                with p.open("rb") as f:
                    f.seek(tail_offset)
                    new_data = f.read()
                    if new_data:
                        buffer.extend(new_data)
                    consumed = 0
                    made = False
                    while True:
                        try:
                            val, used = _decode_varint_from_buffer(buffer, consumed)
                        except ValueError as e:
                            _warn_once(f"Malformed varint in pings stream: {e}")
                            consumed += 1
                            continue
                        if val is None:
                            break
                        length = val
                        if consumed + used + length > len(buffer):
                            break
                        start = consumed + used
                        end = start + length
                        blob = bytes(buffer[start:end])
                        consumed = end
                        msg = salute_pb2.SalutePing()
                        try:
                            msg.ParseFromString(blob)
                            yield msg
                            made = True
                        except Exception as e:
                            _warn_once(f"Decode error for ping: {e}")
                            continue
                    if consumed:
                        del buffer[:consumed]
                        tail_offset += consumed
                    if not made and not new_data:
                        time.sleep(0.2)
            except FileNotFoundError:
                time.sleep(0.2)
                continue

    return _gen()


# ------------------------------- streaming -----------------------------------

def _append_jsonl(out_fp, objs: Iterable[Dict[str, Any]]) -> int:
    n = 0
    for obj in objs:
        out_fp.write(json.dumps(obj) + "\n")
        n += 1
    out_fp.flush()
    os.fsync(out_fp.fileno())
    return n


def stream_feed(
    pings_path: Optional[str | Path] = None,
    events_path: Optional[str | Path] = None,
    out_path: str | Path = "out/c2_feed.jsonl",
    follow: bool = False,
    poll_interval: float = 0.2,
    once: bool = False,
    truncate: bool = False,
    dedup: bool = False,
    dedup_by: str = "event_id",
) -> None:
    seen_keys: set[str] = set()
    """Stream-normalize pings/events into a JSONL sink.

    - Pings: varint length-delimited SALUTE pings.
    - Events: JSONL dicts with required keys.
    - On start, process backlog then (optionally) tail.
    - Deduplication: if `dedup` is True, uses `dedup_by` which can be
      'event_id' (default; key = f"{uid_int}#{ms}") or 'time' (key = f"t#{ms}").
    """

    if not pings_path and not events_path:
        raise ValueError("At least one of --pings or --events is required")

    # Effective mode selection
    mode_follow = bool(follow) and not bool(once)
    mode_once = bool(once) or not bool(follow)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Info on start
    srcs = []
    if events_path:
        srcs.append(f"events={events_path}")
    if pings_path:
        srcs.append(f"pings={pings_path}")
    LOG.info(
        "mock_c2 start: %s; mode=%s; truncate=%s; dedup=%s",
        ", ".join(srcs),
        "follow" if mode_follow and not mode_once else "once",
        bool(truncate),
        bool(dedup),
    )


    mode = "w" if truncate else "a"
    with Path(out_path).open(mode, encoding="utf-8") as out_fp:
        events_p: Optional[Path] = Path(events_path) if events_path else None
        pings_p: Optional[Path] = Path(pings_path) if pings_path else None
        if events_p is not None:
            events_p.parent.mkdir(parents=True, exist_ok=True)
        if pings_p is not None:
            pings_p.parent.mkdir(parents=True, exist_ok=True)

        # Backlog phase
        batch: list[Dict[str, Any]] = []

        # Events backlog
        pos_events = 0
        if events_p is not None and events_p.exists():
            with events_p.open("r", encoding="utf-8") as fp:
                fp.seek(0)
                while True:
                    prev = fp.tell()
                    line = fp.readline()
                    if not line:
                        break
                    if not line.endswith("\n"):
                        fp.seek(prev)
                        break
                    try:
                        raw = json.loads(line)
                    except json.JSONDecodeError as e:
                        _warn_once(f"Skipping malformed JSON line: {e}")
                        pos_events = fp.tell()
                        continue
                    if not isinstance(raw, dict):
                        _warn_once("Skipping malformed JSON line: not an object")
                        pos_events = fp.tell()
                        continue
                    norm = normalize_event_dict(raw)
                    if norm is not None:
                        _maybe_enqueue(norm, batch, dedup, seen_keys, dedup_by)
                    pos_events = fp.tell()

        # Pings backlog
        pos_pings = 0
        if pings_p is not None and pings_p.exists():
            with pings_p.open("rb") as fp:
                fp.seek(0)
                try:
                    for blob in read_ld_stream(fp):
                        pos_pings = fp.tell()
                        try:
                            d = decode_ping(blob)
                            norm = _normalize_from_ping_dict(d)
                            _maybe_enqueue(norm, batch, dedup, seen_keys, dedup_by)
                            try:
                                _write_ping_sizes(len(blob))
                            except Exception:
                                pass
                        except Exception as e:
                            _warn_once(f"Decode error for ping: {e}")
                except ValueError:
                    try:
                        fp.seek(pos_pings)
                    except Exception:
                        pass

        if batch:
            _append_jsonl(out_fp, batch)
            batch = []

        if mode_once and not mode_follow:
            return

        # Tail phase
        ping_buffer = bytearray()
        while True:
            made_progress = False

            # Tail events
            if events_p is not None and events_p.exists():
                with events_p.open("r", encoding="utf-8") as fp:
                    fp.seek(pos_events)
                    while True:
                        prev = fp.tell()
                        line = fp.readline()
                        if not line:
                            break
                        if not line.endswith("\n"):
                            fp.seek(prev)
                            break
                        try:
                            raw = json.loads(line)
                        except json.JSONDecodeError as e:
                            _warn_once(f"Skipping malformed JSON line: {e}")
                            pos_events = fp.tell()
                            continue
                        if not isinstance(raw, dict):
                            _warn_once("Skipping malformed JSON line: not an object")
                            pos_events = fp.tell()
                            continue
                        norm = normalize_event_dict(raw)
                        if norm is not None:
                            _maybe_enqueue(norm, batch, dedup, seen_keys, dedup_by)
                            made_progress = True
                        pos_events = fp.tell()

            # Tail pings
            if pings_p is not None and pings_p.exists():
                with pings_p.open("rb") as fp:
                    fp.seek(pos_pings)
                    new_bytes = fp.read()
                    if new_bytes:
                        ping_buffer.extend(new_bytes)
                    consumed = 0
                    while True:
                        try:
                            length, used = _decode_varint_from_buffer(ping_buffer, consumed)
                        except ValueError as e:
                            _warn_once(f"Malformed varint in pings stream: {e}")
                            consumed += 1
                            continue
                        if length is None:
                            break
                        start = consumed + used
                        end = start + length
                        if end > len(ping_buffer):
                            break
                        blob = bytes(ping_buffer[start:end])
                        consumed = end
                        try:
                            d = decode_ping(blob)
                            norm = _normalize_from_ping_dict(d)
                            _maybe_enqueue(norm, batch, dedup, seen_keys, dedup_by)
                            made_progress = True
                            try:
                                _write_ping_sizes(len(blob))
                            except Exception:
                                pass
                        except Exception as e:
                            _warn_once(f"Decode error for ping: {e}")
                            continue
                    if consumed:
                        del ping_buffer[:consumed]
                        pos_pings += consumed

            if batch:
                _append_jsonl(out_fp, batch)
                batch = []

            if not mode_follow:
                break

            if not made_progress:
                time.sleep(poll_interval)


# ------------------------------- CLI -----------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Mock C2 feed normalizer")
    p.add_argument("--pings", type=str, default=None, help="Path to length-delimited pings.bin")
    p.add_argument("--events", type=str, default=None, help="Path to events.jsonl")
    p.add_argument("--out", type=str, default="out/c2_feed.jsonl", help="Output JSONL file")
    p.add_argument("--follow", action="store_true", help="Tail the inputs for new data")
    p.add_argument("--once", action="store_true", help="Process backlog once then exit")
    p.add_argument("--poll", type=float, default=0.2, help="Follow poll interval seconds")
    p.add_argument("--truncate", action="store_true", help="Truncate output file on start (do not append)")
    p.add_argument("--dedup", action="store_true", help="Skip duplicates by (uid,ts_ns) within this run")
    p.add_argument("--log-level", type=str, default="INFO", help="Logging level (e.g., INFO, DEBUG)")
    p.add_argument("--dedup-by", type=str, default="event_id",
               choices=["event_id", "time"],
               help="Deduplicate by 'event_id' (uid#ms, default) or by 'time' (ms suffix only)")
    return p

#공동 키 생성기
def _dedup_key(obj: Dict[str, Any], mode: str) -> Optional[str]:
    try:
        if mode == "event_id":
            return str(obj.get("event_id"))
        elif mode == "time":
            # uid 무시, 동일 ms 타임스탬프만 기준
            ts_ns = int(obj["ts_ns"])
            suffix_ms = int(round(ts_ns / 1_000_000.0))
            return f"t#{suffix_ms}"
    except Exception:
        return None
    return None

# 전역 세트 하나만 쓰면 모드 바꿀 때 섞일 수 있으니, 세트는 호출부에서 주입
def _maybe_enqueue(obj: Dict[str, Any], batch: list[Dict[str, Any]],
                   dedup: bool, seen: set[str], dedup_by: str) -> None:
    if not dedup:
        batch.append(obj); return
    key = _dedup_key(obj, dedup_by)
    if key is None or key not in seen:
        if key is not None:
            seen.add(key)
        batch.append(obj)

# pings 정규화 (_normalize_from_ping_dict)
def _normalize_from_ping_dict(d: Dict[str, Any]) -> Dict[str, Any]:
    uid_int = _uid_to_int(d["uid"])       # ← 정수 강제
    ts_ns = int(d["ts_ns"])
    lat_q = int(d["lat_q"])
    lon_q = int(d["lon_q"])
    event_code = int(d["event_code"])

    lat = float(lat_q) / 1e7
    lon = float(lon_q) / 1e7
    suffix_ms = _suffix_ms_from_ts(ts_ns) # ← ms 접미사 통일

    return {
        "event_id": f"{uid_int}#{suffix_ms}",
        "uid": uid_int,                    # ← 항상 정수
        "ts_ns": ts_ns,
        "lat": lat,
        "lon": lon,
        "event_code": event_code,
        "conf": 0.5,
        "intent_score": 0.0,
    }

def main(argv: Optional[list[str]] = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(level=getattr(logging, str(args.log_level).upper(), logging.INFO))

    try:
        stream_feed(
            pings_path=args.pings,
            events_path=args.events,
            out_path=args.out,
            follow=bool(args.follow),
            poll_interval=float(args.poll),
            once=bool(args.once),
            truncate=bool(args.truncate),
            dedup=bool(args.dedup),
            dedup_by=str(args.dedup_by),
        )
        return 0
    except Exception as e:
        LOG.error("mock_c2 failed: %s", e)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
