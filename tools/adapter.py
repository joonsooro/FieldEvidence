#!/usr/bin/env python3
"""
Adapter CLI

Maps vendor JSON/CSV lines into canonical JSONL records so the existing C2
loop can be reused as-is.

Canonical schema per line (exact keys):
{"ts_ns": int, "lat": float, "lon": float, "intent_score": float, "uid": <int|str>, "event_code": int}

Profiles:
- simple_json  – flat JSONL (e.g., {"ts_ms":..., "lat":..., "lon":..., "intent":..., "id":..., "code":...})
- nested_json  – nested JSONL (e.g., {"t": "...ISO8601...", "pos":{"lat":...,"lon":...}, "scores":{"intent":...}, "uid":..., "event":{ "code":... }})
- csv_v1       – CSV with header ts_ms,lat,lon,uid,intent,event_code

Robust parsing rules:
- Timestamp mapping:
  Prefer ts_ns; else ts_ms * 1_000_000; else ts_s * 1_000_000_000; else parse ISO8601 string (UTC) to ns.
- Latitude/Longitude must be numeric and valid ranges (lat ∈ [-90,90], lon ∈ [-180,180]) or skip.
- intent_score default 0.0 if missing; clamp to [0,1].
- uid can be string or int; do not hash here—just pass through.
- event_code default 1 if missing.

Behavior:
- Invalid lines are skipped with a concise WARNING to stderr; the tool never crashes on bad input.
- Output is JSONL; one canonical record per valid input line/row.

CLI:
- --in (path or - for stdin), --out (path, default out/events.jsonl), --profile {simple_json,nested_json,csv_v1}
- --limit (optional int) to stop after N valid outputs (useful for demos)
- --dry-run prints to stdout instead of writing a file
- --verbose enables per-line diagnostics

Implementation notes:
- Keep mapping functions small and focused: map_simple_json(d), map_nested_json(d), map_csv_row(row).
- Use a tiny helper to_ts_ns(obj) that recognizes ts_ns, ts_ms, ts_s, and ISO8601.
- Use argparse, json, csv, datetime, dateutil (if unavailable, fallback to fromisoformat with replace("Z","+00:00")).
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import sys
from typing import Any, Dict, Optional, Tuple, Union

from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP

try:
    # Prefer python-dateutil if available
    from dateutil import parser as dateutil_parser  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    dateutil_parser = None  # type: ignore


def warn(msg: str) -> None:
    sys.stderr.write(f"WARNING adapter: {msg}\n")


def info(msg: str) -> None:
    # Summary/info messages
    sys.stderr.write(f"adapter: {msg}\n")


def clamp01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


def _to_float(v: Any) -> Optional[float]:
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        v = v.strip()
        if not v:
            return None
        try:
            return float(v)
        except Exception:
            return None
    return None


def _to_int(v: Any) -> Optional[int]:
    if v is None:
        return None
    if isinstance(v, bool):  # avoid True/False becoming 1/0
        return None
    if isinstance(v, int):
        return v
    if isinstance(v, float):
        try:
            return int(v)
        except Exception:
            return None
    if isinstance(v, str):
        s = v.strip()
        if not s:
            return None
        try:
            # Support integer-like strings and floats like "2.0"
            if "." in s:
                return int(float(s))
            return int(s)
        except Exception:
            return None
    return None


def parse_iso_to_ns(s: str) -> Optional[int]:
    # Parse ISO8601 to epoch ns (UTC). Try dateutil if available; fallback to fromisoformat.
    if not isinstance(s, str) or not s:
        return None
    try:
        if dateutil_parser is not None:
            dt = dateutil_parser.isoparse(s)
        else:
            s2 = s.replace("Z", "+00:00")
            dt = datetime.fromisoformat(s2)
    except Exception:
        return None
    try:
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        else:
            dt = dt.astimezone(timezone.utc)
        # Convert to ns; using float timestamp is acceptable for second precision inputs
        return int(dt.timestamp() * 1_000_000_000)
    except Exception:
        return None


def _ms_to_ns_int(ts_ms: Any) -> Optional[int]:
    """Convert milliseconds to nanoseconds using integer/decimal math only.

    Accepts int/str/float. Prefers integer coercion to avoid precision loss.
    Falls back to Decimal for float inputs to preserve large magnitudes.
    """
    # Fast path: integers and integer-like strings
    iv = _to_int(ts_ms)
    if iv is not None:
        try:
            return int(iv) * 1_000_000
        except Exception:
            return None
    # Float path: use Decimal for large magnitudes to avoid binary FP drift
    if isinstance(ts_ms, float):
        try:
            return int((Decimal(str(ts_ms)) * Decimal(1_000_000)).to_integral_value(ROUND_HALF_UP))
        except Exception:
            return None
    # Last resort: string or other types coerced to string
    try:
        return int(str(ts_ms).strip()) * 1_000_000
    except Exception:
        return None


def to_ts_ns(obj: Any) -> Optional[int]:
    """Convert multiple timestamp representations to epoch nanoseconds.

    Behavior:
    - If obj is dict: prefer obj["ts_ns"]; else obj["ts_ms"]*1e6; else obj["ts_s"]*1e9; else obj["t"] ISO8601.
    - If obj is str: try ISO8601 to ns.
    """
    # Dict handling with precedence
    if isinstance(obj, dict):
        if "ts_ns" in obj:
            v = _to_int(obj.get("ts_ns"))
            return v
        if "ts_ms" in obj:
            ns = _ms_to_ns_int(obj.get("ts_ms"))
            if ns is not None:
                return ns
            # fallthrough if still None

        if "ts_s" in obj:
            # Prefer integer seconds path
            iv = _to_int(obj.get("ts_s"))
            if iv is not None:
                try:
                    return int(iv) * 1_000_000_000
                except Exception:
                    return None
            # Support fractional seconds using Decimal
            fv = _to_float(obj.get("ts_s"))
            if fv is not None:
                try:
                    return int((Decimal(str(fv)) * Decimal(1_000_000_000)).to_integral_value(ROUND_HALF_UP))
                except Exception:
                    return None
        # Common nested key: ISO string under 't'
        if "t" in obj:
            return parse_iso_to_ns(obj.get("t"))
        return None
    # String handling: ISO8601
    if isinstance(obj, str):
        return parse_iso_to_ns(obj)
    return None


def validate_lat_lon(lat: Any, lon: Any) -> Optional[Tuple[float, float]]:
    """Validate and coerce lat/lon. Returns (lat, lon) or None if invalid."""
    flat = _to_float(lat)
    flon = _to_float(lon)
    if flat is None or flon is None:
        return None
    if not (-90.0 <= flat <= 90.0):
        return None
    if not (-180.0 <= flon <= 180.0):
        return None
    return flat, flon


def canonical_record(ts_ns: int, lat: float, lon: float, intent_score: float, uid: Union[str, int], event_code: int) -> Dict[str, Any]:
    """Build canonical record with exact keys in stable order."""
    # Python 3.7+ preserves insertion order
    rec: Dict[str, Any] = {}
    rec["ts_ns"] = int(ts_ns)
    rec["lat"] = float(lat)
    rec["lon"] = float(lon)
    rec["intent_score"] = float(intent_score)
    rec["uid"] = uid
    rec["event_code"] = int(event_code)
    return rec


def map_simple_json(d: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Map flat JSON to canonical schema.

    Expected fields:
    - ts_ns|ts_ms|ts_s
    - lat, lon
    - intent -> intent_score (default 0.0; clamp)
    - id -> uid
    - code -> event_code (default 1)
    """
    ts = to_ts_ns(d)
    if ts is None:
        return None, "missing/invalid timestamp"
    coords = validate_lat_lon(d.get("lat"), d.get("lon"))
    if coords is None:
        return None, "invalid lat/lon"
    lat, lon = coords
    intent = _to_float(d.get("intent"))
    if intent is None:
        intent = 0.0
    intent = clamp01(intent)
    uid = d.get("id")
    if uid is None or (isinstance(uid, str) and uid.strip() == ""):
        return None, "missing uid"
    # Keep uid as int or str; coerce other types to str
    if not isinstance(uid, (str, int)):
        uid = str(uid)
    event_code = d.get("code")
    ev = _to_int(event_code)
    if ev is None:
        ev = 1
    return canonical_record(ts, lat, lon, intent, uid, ev), None


def map_nested_json(d: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Map nested JSON to canonical schema.

    Expected fields:
    - t (ISO8601) or ts_ns|ts_ms|ts_s
    - pos.lat, pos.lon
    - scores.intent -> intent_score (default 0.0; clamp)
    - uid
    - event.code -> event_code (default 1)
    """
    ts = to_ts_ns(d)
    if ts is None:
        return None, "missing/invalid timestamp"
    pos = d.get("pos") or {}
    coords = validate_lat_lon(pos.get("lat"), pos.get("lon"))
    if coords is None:
        return None, "invalid lat/lon"
    lat, lon = coords
    scores = d.get("scores") or {}
    intent = _to_float(scores.get("intent"))
    if intent is None:
        intent = 0.0
    intent = clamp01(intent)
    uid = d.get("uid")
    if uid is None or (isinstance(uid, str) and uid.strip() == ""):
        return None, "missing uid"
    if not isinstance(uid, (str, int)):
        uid = str(uid)
    event = d.get("event") or {}
    ev = _to_int(event.get("code"))
    if ev is None:
        ev = 1
    return canonical_record(ts, lat, lon, intent, uid, ev), None


def map_csv_row(row: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Map CSV row to canonical schema.

    Expected header: ts_ms,lat,lon,uid,intent,event_code
    """
    ts = to_ts_ns({"ts_ms": row.get("ts_ms")})
    if ts is None:
        return None, "missing/invalid timestamp"
    coords = validate_lat_lon(row.get("lat"), row.get("lon"))
    if coords is None:
        return None, "invalid lat/lon"
    lat, lon = coords
    intent = _to_float(row.get("intent"))
    if intent is None:
        intent = 0.0
    intent = clamp01(intent)
    uid = row.get("uid")
    if uid is None or (isinstance(uid, str) and uid.strip() == ""):
        return None, "missing uid"
    # CSV values are strings; keep uid as str
    event_code = row.get("event_code")
    ev = _to_int(event_code)
    if ev is None:
        ev = 1
    return canonical_record(ts, lat, lon, intent, uid, ev), None


def iter_jsonl_lines(stream: io.TextIOBase):
    for line_no, line in enumerate(stream, start=1):
        s = line.strip()
        if not s:
            continue
        try:
            obj = json.loads(s)
        except Exception:
            yield line_no, None, "invalid JSON"
            continue
        if not isinstance(obj, dict):
            yield line_no, None, "not an object"
            continue
        yield line_no, obj, None


def iter_csv_rows(stream: io.TextIOBase):
    reader = csv.DictReader(stream)
    for line_no, row in enumerate(reader, start=2):  # header is line 1
        yield line_no, row


def main() -> None:
    parser = argparse.ArgumentParser(description="Adapt vendor inputs to canonical JSONL")
    parser.add_argument("--in", dest="inp", required=True, help="Input path or - for stdin")
    parser.add_argument("--out", dest="out", default="out/events.jsonl", help="Output path (append)")
    parser.add_argument("--profile", choices=["simple_json", "nested_json", "csv_v1"], required=True)
    parser.add_argument("--limit", type=int, default=None, help="Stop after N valid outputs")
    parser.add_argument("--dry-run", action="store_true", help="Print to stdout instead of writing a file")
    parser.add_argument("--verbose", action="store_true", help="Enable per-line diagnostics")
    parser.add_argument("--truncate-out", action="store_true",
                        help="Truncate the output file before writing (takes precedence over --out-mode)")
    parser.add_argument("--out-mode", choices=["append", "overwrite"], default="append",
                        help="Output file mode: append (default) or overwrite (open with 'w')")

    args = parser.parse_args()

    profile = args.profile
    valid = 0
    skipped = 0

    # Prepare IO
    inp_stream: io.TextIOBase
    close_in = False
    try:
        if args.inp == "-":
            inp_stream = sys.stdin
        else:
            inp_stream = open(args.inp, "r", encoding="utf-8")
            close_in = True
    except Exception as e:
        warn(f"cannot open input: {e}")
        sys.exit(2)

    out_stream: Optional[io.TextIOBase] = None
    close_out = False
    try:
        if not args.dry_run:
            out_dir = os.path.dirname(args.out)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
            # Determine output mode: --truncate-out takes precedence
            mode = "w" if args.truncate_out or args.out_mode == "overwrite" else "a"
            out_stream = open(args.out, mode, encoding="utf-8")
            close_out = True
            if args.verbose:
                info(f"open '{args.out}' with mode='{mode}'")
        else:
            out_stream = None
    except Exception as e:
        if close_in:
            inp_stream.close()
        warn(f"cannot open output: {e}")
        sys.exit(2)

    def emit(rec: Dict[str, Any]) -> None:
        s = json.dumps(rec, ensure_ascii=False, separators=(",", ":"))
        if args.dry_run:
            print(s)
        else:
            assert out_stream is not None
            out_stream.write(s + "\n")

    try:
        if profile in ("simple_json", "nested_json"):
            for line_no, obj, err in iter_jsonl_lines(inp_stream):
                if err is not None:
                    skipped += 1
                    if args.verbose:
                        warn(f"line {line_no} skipped: {err}")
                    continue
                assert obj is not None
                if profile == "simple_json":
                    rec, reason = map_simple_json(obj)
                else:
                    rec, reason = map_nested_json(obj)
                if rec is None:
                    skipped += 1
                    if args.verbose:
                        warn(f"line {line_no} skipped: {reason}")
                    continue
                emit(rec)
                valid += 1
                if args.limit is not None and valid >= args.limit:
                    break
        else:  # csv_v1
            try:
                for line_no, row in iter_csv_rows(inp_stream):
                    rec, reason = map_csv_row(row)
                    if rec is None:
                        skipped += 1
                        if args.verbose:
                            warn(f"line {line_no} skipped: {reason}")
                        continue
                    emit(rec)
                    valid += 1
                    if args.limit is not None and valid >= args.limit:
                        break
            except csv.Error as e:
                warn(f"CSV parse error: {e}")
    finally:
        if close_in:
            inp_stream.close()
        if close_out and out_stream is not None:
            out_stream.flush()
            out_stream.close()

    info(f"wrote {valid} record(s); skipped {skipped} invalid line(s)")


if __name__ == "__main__":
    main()
