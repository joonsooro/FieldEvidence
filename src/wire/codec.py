"""SALUTE microping codec utilities."""
from __future__ import annotations

from typing import Any, BinaryIO, Dict, Generator, Iterable

from google.protobuf.message import DecodeError

from . import salute_pb2

_REQUIRED_FIELDS = {
    "uid": int,
    "ts_ns": int,
    "lat_q": int,
    "lon_q": int,
    "geo_conf_m": int,
    "event_code": int,
    "hash_pref": (bytes, bytearray, memoryview),
}


def _coerce_hash_pref(value: Any) -> bytes:
    if isinstance(value, bytes):
        return value
    if isinstance(value, bytearray):
        return bytes(value)
    if isinstance(value, memoryview):
        return value.tobytes()
    raise ValueError("hash_pref must be bytes-like")


def _validate_fields(data: Dict[str, Any]) -> Dict[str, Any]:
    for field, expected_type in _REQUIRED_FIELDS.items():
        if field not in data:
            raise ValueError(f"Missing required field: {field}")
        value = data[field]
        if field == "hash_pref":
            hashed = _coerce_hash_pref(value)
            if not (4 <= len(hashed) <= 8):
                raise ValueError("hash_pref must be 4-8 bytes")
            data[field] = hashed
            continue
        if not isinstance(value, expected_type):
            raise ValueError(f"Field '{field}' must be {expected_type.__name__}")
    event_code = data["event_code"]
    if event_code not in {1, 2, 3}:
        raise ValueError("event_code must be 1, 2, or 3")
    return data


def encode_ping(payload: Dict[str, Any]) -> bytes:
    """Serialize a SALUTE ping dictionary to bytes."""
    validated = _validate_fields(dict(payload))
    message = salute_pb2.SalutePing(
        uid=validated["uid"],
        ts_ns=validated["ts_ns"],
        lat_q=validated["lat_q"],
        lon_q=validated["lon_q"],
        geo_conf_m=validated["geo_conf_m"],
        event_code=validated["event_code"],
        hash_pref=validated["hash_pref"],
    )
    return message.SerializeToString()


def decode_ping(blob: bytes) -> Dict[str, Any]:
    """Deserialize bytes into a SALUTE ping dictionary."""
    if not isinstance(blob, (bytes, bytearray, memoryview)):
        raise ValueError("Input must be bytes-like")
    message = salute_pb2.SalutePing()
    try:
        message.ParseFromString(bytes(blob))
    except DecodeError as exc:
        raise ValueError("Failed to decode SALUTE ping") from exc
    return {
        "uid": message.uid,
        "ts_ns": message.ts_ns,
        "lat_q": message.lat_q,
        "lon_q": message.lon_q,
        "geo_conf_m": message.geo_conf_m,
        "event_code": message.event_code,
        "hash_pref": bytes(message.hash_pref),
    }


def estimate_size(payload: Dict[str, Any]) -> int:
    """Return the encoded size of a payload in bytes."""
    return len(encode_ping(payload))


def _encode_varint(value: int) -> bytes:
    if value < 0:
        raise ValueError("Length must be non-negative")
    out = bytearray()
    while True:
        to_write = value & 0x7F
        value >>= 7
        if value:
            out.append(to_write | 0x80)
        else:
            out.append(to_write)
            break
    return bytes(out)


def _decode_varint(stream: BinaryIO) -> int | None:
    shift = 0
    result = 0
    while True:
        byte = stream.read(1)
        if byte == b"":
            if shift == 0:
                return None
            raise EOFError("Unexpected EOF while reading length")
        b = byte[0]
        result |= (b & 0x7F) << shift
        if not b & 0x80:
            return result
        shift += 7
        if shift >= 64:
            raise ValueError("Varint length exceeds 64 bits")


def write_ld_stream(messages: Iterable[bytes], fp: BinaryIO) -> None:
    """Write length-delimited messages to a binary stream."""
    for blob in messages:
        if not isinstance(blob, (bytes, bytearray, memoryview)):
            raise ValueError("Stream payloads must be bytes-like")
        payload = bytes(blob)
        fp.write(_encode_varint(len(payload)))
        fp.write(payload)


def read_ld_stream(fp: BinaryIO) -> Generator[bytes, None, None]:
    """Yield length-delimited message blobs from a binary stream."""
    while True:
        try:
            length = _decode_varint(fp)
        except EOFError as exc:
            raise ValueError("Truncated length-delimited stream") from exc
        if length is None:
            return
        data = fp.read(length)
        if len(data) != length:
            raise ValueError("Truncated message payload")
        yield data
