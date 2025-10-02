from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

from src.wire import salute_pb2


ISO_FMT = "%Y-%m-%dT%H:%M:%S.%fZ"


def _iso_utc_from_ns(ts_ns: int) -> str:
    # Convert epoch nanoseconds to UTC ISO8601 with 'Z'.
    # Truncate to microseconds for XML readability/compat.
    sec = ts_ns / 1e9
    dt = datetime.fromtimestamp(sec, tz=timezone.utc)
    # Normalize microseconds formatting (strip trailing zeros)
    s = dt.strftime(ISO_FMT)
    # Reduce excessive trailing zeros in fractional seconds
    # e.g., 2024-01-01T00:00:00.000000Z -> 2024-01-01T00:00:00Z
    if ".000000Z" in s:
        s = s.replace(".000000Z", "Z")
    return s


def microping_to_cot_xml(ping: salute_pb2.SalutePing, callsign: str = "UAV-1") -> str:
    """
    Convert a SALUTE microping to a minimal CoT event XML string.

    - type="a-f-A-M-F" (constant)
    - uid="uid_<uint64>"
    - time/start from ts_ns; stale = start + 30s
    - how="m-g"
    - point: lat={lat_q/1e7} lon={lon_q/1e7} hae=0 ce=50 le=50
    - detail: <contact callsign=.../> and <remarks>event_code=... hash_len=...</remarks>
    """
    # Coerce/compute fields
    uid = int(ping.uid)
    uid_s = f"uid_{uid}"
    ts_ns = int(ping.ts_ns)
    time_s = _iso_utc_from_ns(ts_ns)
    stale_ns = ts_ns + int(30 * 1e9)
    stale_s = _iso_utc_from_ns(stale_ns)

    lat = float(ping.lat_q) / 1e7
    lon = float(ping.lon_q) / 1e7
    ce = 50
    le = 50
    hae = 0

    event_code = int(ping.event_code)
    hash_len = len(bytes(ping.hash_pref))

    # Construct minimal CoT XML
    # Note: Keep attributes ordered for readability (not required by parser)
    xml = (
        f"<event version=\"2.0\" type=\"a-f-A-M-F\" uid=\"{uid_s}\" "
        f"time=\"{time_s}\" start=\"{time_s}\" stale=\"{stale_s}\" how=\"m-g\">"
        f"<point lat=\"{lat:.7f}\" lon=\"{lon:.7f}\" hae=\"{hae}\" ce=\"{ce}\" le=\"{le}\"/>"
        f"<detail>"
        f"<contact callsign=\"{callsign}\"/>"
        f"<remarks>event_code={event_code} hash_len={hash_len}</remarks>"
        f"</detail>"
        f"</event>"
    )
    return xml


def write_cot_file(xml: str, out_dir: Path, uid: int, ts_ns: int) -> Path:
    """Write a CoT XML string to `out_dir/uid_<uid>_<tsns>.xml` and return the path."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"uid_{int(uid)}_{int(ts_ns)}.xml"
    with path.open("w", encoding="utf-8") as f:
        f.write(xml)
    return path

