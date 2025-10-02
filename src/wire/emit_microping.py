from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from . import salute_pb2
from .codec import write_ld_stream


MAX_PAYLOAD = 80  # bytes, protobuf payload only (excludes varint length framing)


@dataclass
class Ping:
    payload: bytes  # protobuf-serialized SalutePing
    size: int       # len(payload)
    meta: Dict[str, Any]


def _warn(msg: str) -> None:
    sys.stderr.write(f"[warn] {msg}\n")
    sys.stderr.flush()


def _err(msg: str) -> None:
    sys.stderr.write(f"[error] {msg}\n")
    sys.stderr.flush()


def _sha256(data: bytes) -> bytes:
    return hashlib.sha256(data).digest()


def _hash_uid(uid: Any, frame: Any) -> int:
    s = f"{uid}:{frame}".encode("utf-8")
    h = int.from_bytes(_sha256(s), byteorder="big", signed=False)
    return h & ((1 << 64) - 1)


def _canonical_json_bytes(obj: Dict[str, Any]) -> bytes:
    # Deterministic JSON for hashing
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _q32(v: float) -> int:
    # Quantize to signed 32-bit of 1e-7 degrees
    q = int(round(float(v) * 1e7))
    # Clamp to int32 range
    if q < -2_147_483_648:
        q = -2_147_483_648
    elif q > 2_147_483_647:
        q = 2_147_483_647
    return q


def _build_ping(ev: Dict[str, Any], hash_pref_len: int) -> Tuple[bytes, Dict[str, Any]]:
    # Validate minimal fields
    required = ("uid", "frame", "ts_ns", "lat", "lon", "event_code")
    missing = [k for k in required if k not in ev]
    if missing:
        raise KeyError(f"missing fields: {','.join(missing)}")

    uid64 = _hash_uid(ev["uid"], ev["frame"])
    ts_ns = int(ev["ts_ns"])
    lat_q = _q32(float(ev["lat"]))
    lon_q = _q32(float(ev["lon"]))
    geo_conf_m = 50
    event_code = int(ev["event_code"])  # passthrough

    # Hash prefix over canonicalized event JSON
    h_full = _sha256(_canonical_json_bytes(ev))
    hash_pref = h_full[:hash_pref_len]

    msg = salute_pb2.SalutePing(
        uid=uid64,
        ts_ns=ts_ns,
        lat_q=lat_q,
        lon_q=lon_q,
        geo_conf_m=geo_conf_m,
        event_code=event_code,
        hash_pref=hash_pref,
    )
    blob = msg.SerializeToString()
    meta = {
        "uid64": uid64,
        "ts_ns": ts_ns,
        "lat_q": lat_q,
        "lon_q": lon_q,
        "event_code": event_code,
    }
    return blob, meta


def _load_events(path: Path) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if not isinstance(obj, dict):
                    _warn(f"line {i}: not an object; skipped")
                    continue
                events.append(obj)
            except Exception as e:
                _warn(f"line {i}: invalid JSON; skipped ({e})")
    return events


def _encode_all(events: List[Dict[str, Any]], hash_pref_len: int) -> Tuple[List[Ping], List[Tuple[int, Dict[str, Any], str]]]:
    pings: List[Ping] = []
    bad: List[Tuple[int, Dict[str, Any], str]] = []  # (index, event, reason)
    for idx, ev in enumerate(events):
        try:
            blob, meta = _build_ping(ev, hash_pref_len)
            pings.append(Ping(payload=blob, size=len(blob), meta=meta))
        except KeyError as e:
            bad.append((idx, ev, str(e)))
        except Exception as e:
            bad.append((idx, ev, f"encode_error: {e}"))
    return pings, bad


def _percentile(sorted_values: List[int], p: float) -> int:
    if not sorted_values:
        return 0
    if p <= 0:
        return sorted_values[0]
    if p >= 1:
        return sorted_values[-1]
    k = int(round((len(sorted_values) - 1) * p))
    return sorted_values[k]


def _write_stream(pings: Iterable[Ping], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as f:
        write_ld_stream((p.payload for p in pings), f)


def _report_sizes(pings: List[Ping], hash_pref_len: int, report_path: Path | None) -> Dict[str, Any]:
    sizes = sorted(p.size for p in pings)
    count = len(sizes)
    avg = (sum(sizes) / count) if count else 0.0
    p50 = _percentile(sizes, 0.50)
    p95 = _percentile(sizes, 0.95)
    maxv = sizes[-1] if sizes else 0
    # Console summary
    print(f"microping sizes — count={count} avg={avg:.1f} p50={p50} p95={p95} max={maxv} (hash_pref={hash_pref_len})")
    # JSON report
    report = {
        "count": count,
        "avg": round(avg, 1),
        "p50": p50,
        "p95": p95,
        "max": maxv,
        "hash_pref_bytes": hash_pref_len,
    }
    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with report_path.open("w", encoding="utf-8") as rf:
            json.dump(report, rf)
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Convert detection events to SALUTE micropings (≤80B)")
    p.add_argument("--events", required=True, help="JSONL input (one event per line)")
    p.add_argument("--out", required=True, help="Output varint-framed binary stream")
    p.add_argument("--size_report", required=True, help="JSON file for size statistics")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    events_path = Path(args.events)
    out_path = Path(args.out)
    report_path = Path(args.size_report)

    if not events_path.exists():
        _err(f"events not found: {events_path}")
        raise SystemExit(1)

    try:
        events = _load_events(events_path)
    except Exception as e:
        _err(f"failed to read events: {e}")
        raise SystemExit(1)

    if not events:
        _err("no events in input")
        raise SystemExit(1)

    # First pass with 6-byte hash prefix
    pings6, bad6 = _encode_all(events, hash_pref_len=6)
    for idx, _, reason in bad6:
        _warn(f"event {idx}: {reason}")

    if not pings6:
        _err("no valid events after filtering")
        raise SystemExit(1)

    max6 = max(p.size for p in pings6)
    if max6 <= MAX_PAYLOAD:
        # Write and report
        try:
            _write_stream(pings6, out_path)
        except Exception as e:
            _err(f"failed to write output: {e}")
            raise SystemExit(1)
        _ = _report_sizes(pings6, 6, report_path)
        return

    # Fallback to 4-byte hash prefix
    pings4, bad4 = _encode_all(events, hash_pref_len=4)
    for idx, _, reason in bad4:
        _warn(f"event {idx}: {reason}")

    if not pings4:
        _err("no valid events after fallback encoding")
        raise SystemExit(1)

    max4 = max(p.size for p in pings4)
    if max4 <= MAX_PAYLOAD:
        try:
            _write_stream(pings4, out_path)
        except Exception as e:
            _err(f"failed to write output: {e}")
            raise SystemExit(1)
        _ = _report_sizes(pings4, 4, report_path)
        return

    # Still oversize → print top 5 and exit 2
    overs = sorted(((p.size, i, p) for i, p in enumerate(pings4) if p.size > MAX_PAYLOAD), reverse=True)
    _err(f"oversize messages even with 4-byte hash_pref (max={max4} > {MAX_PAYLOAD})")
    for rank, item in enumerate(overs[:5], start=1):
        sz, i, p = item
        meta = p.meta
        _err(f"  #{rank} idx={i} size={sz} uid64={meta.get('uid64')} ts_ns={meta.get('ts_ns')} ec={meta.get('event_code')}")
    raise SystemExit(2)


if __name__ == "__main__":
    main()

