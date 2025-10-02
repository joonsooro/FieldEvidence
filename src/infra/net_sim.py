from __future__ import annotations
from pathlib import Path

_SENT_FILE = Path("out/sent.bin")
_ONLINE: bool = False

def set_online(flag: bool) -> None:
    global _ONLINE
    _ONLINE = bool(flag)

def is_online() -> bool:
    return _ONLINE

def send_http(payload: bytes) -> None:
    if not is_online():
        raise RuntimeError("offline")
    _SENT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with _SENT_FILE.open("ab") as f:
        f.write(payload)
        f.flush()
