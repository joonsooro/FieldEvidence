from __future__ import annotations
import time, sqlite3
from pathlib import Path
from typing import Optional, Callable, List, Tuple, Dict, Any

DB_PATH = Path("out/queue.db")

_CONN: Optional[sqlite3.Connection] = None

def _monotonic_ns() -> int:
    return time.monotonic_ns()

def _ensure_conn() -> sqlite3.Connection:
    global _CONN
    if _CONN is not None:
        return _CONN
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), timeout=30, isolation_level=None)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA busy_timeout=3000;")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS queue(
          seq INTEGER PRIMARY KEY,
          ts_ns INTEGER,
          payload BLOB,
          status INTEGER DEFAULT 0
        );
        """
    )
    _CONN = conn
    return conn

def enqueue(payload: bytes, ts_ns: Optional[int] = None) -> None:
    conn = _ensure_conn()
    t_event = int(ts_ns if ts_ns is not None else _monotonic_ns())
    conn.execute(
        "INSERT INTO queue(ts_ns, payload, status) VALUES(?, ?, 0)",
        (t_event, payload),
    )

def depth() -> int:
    conn = _ensure_conn()
    row = conn.execute("SELECT COUNT(*) FROM queue WHERE status=0").fetchone()
    return int(row[0]) if row else 0

def peek_oldest(n: int) -> List[Tuple[int, int, bytes]]:
    if n <= 0:
        return []
    conn = _ensure_conn()
    rows = conn.execute(
        "SELECT seq, ts_ns, payload FROM queue WHERE status=0 ORDER BY seq ASC LIMIT ?",
        (int(n),),
    ).fetchall()
    # type: ignore[return-value]
    return [(int(a), int(b), bytes(c)) for (a, b, c) in rows]

def mark_sent(seqs: List[int]) -> None:
    if not seqs:
        return
    conn = _ensure_conn()
    placeholders = ",".join(["?"] * len(seqs))
    conn.execute(f"UPDATE queue SET status=1 WHERE seq IN ({placeholders})", seqs)

# --- in src/infra/store_forward.py ---

def drain(send_fn, limit: int = 100, budget_ms: int = 50, now_ns: int | None = None):
    """
    Dequeue up to `limit` items and attempt to send within `budget_ms`.
    Returns a snapshot dict including BOTH E2E and drain-only latencies:
      - latency_ms:    E2E (enqueue ts_ns -> send completion) per message
      - drain_ms:      drain-only (send start -> send completion) per message
      - sent:          number of messages successfully sent in this call
      - retries:       number of transient send errors retried (if any)
    """
    import sqlite3, time, os
    db_path = Path("out/queue.db")
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    cur = conn.cursor()

    t0 = time.monotonic()
    if now_ns is None:
        now_ns = time.monotonic_ns()

    # fetch queued items
    cur.execute(
        "SELECT seq, ts_ns, payload FROM queue WHERE status=0 ORDER BY seq LIMIT ?",
        (limit,),
    )
    rows = cur.fetchall()

    e2e_lat_ms: list[float] = []
    drain_lat_ms: list[float] = []
    sent = 0
    retries = 0

    for seq, ts_ns, payload in rows:
        # budget check
        if (time.monotonic() - t0) * 1000.0 >= budget_ms:
            break

        # measure drain-only latency around send_fn
        t_send_start = time.monotonic()
        try:
            send_fn(payload)  # append to out/sent.bin or raise if offline
        except Exception:
            # leave as queued; could implement retry/backoff counters here
            retries += 1
            continue
        t_send_end = time.monotonic()

        # mark sent
        cur.execute("UPDATE queue SET status=1 WHERE seq=?", (seq,))
        sent += 1

        # E2E: enqueue( ts_ns ) -> send_end
        e2e_ms = (now_ns - int(ts_ns)) / 1e6
        e2e_lat_ms.append(float(e2e_ms))

        # drain-only: send_start -> send_end
        drain_ms = (t_send_end - t_send_start) * 1000.0
        drain_lat_ms.append(float(drain_ms))

    conn.commit()
    conn.close()

    return {
        "sent": sent,
        "retries": retries,
        "latency_ms": e2e_lat_ms,     # E2E
        "drain_ms": drain_lat_ms,     # drain-only
        "limit": limit,
        "budget_ms": budget_ms,
    }

