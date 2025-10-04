from __future__ import annotations
import time, sqlite3
from pathlib import Path
from typing import Optional, Callable, List, Tuple, Dict, Any

from src.infra import paths

DB_PATH = paths.DB_PATH

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

def clear() -> None:
    """
    Truncate/clear the queue table and reclaim space.
    Safe to call repeatedly.
    """
    conn = _ensure_conn()
    try:
        conn.execute("DELETE FROM queue;")
        # VACUUM requires no active transaction; autocommit is enabled
        conn.execute("VACUUM;")
    except Exception:
        # If anything goes wrong, ensure table exists then retry delete only
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
        conn.execute("DELETE FROM queue;")

def drain(
    send_fn,
    limit: int = 100,
    budget_ms: int = 50,
    now_ns: int | None = None,
    require_online: Optional[Callable[[], bool]] = None,
):
    """
    Dequeue up to `limit` items and attempt to send within `budget_ms` using
    a single time base (time.monotonic_ns). Returns a dict with:
      - sent:     number of messages successfully sent in this call
      - retries:  number of transient send errors encountered
      - limit, budget_ms: echo of input params (for diagnostics)
    Latency samples are measured internally for potential future use, but are
    not returned to keep callers' counters simple and stable.
    """
    import sqlite3, time, os
    # Early exit if offline requested by caller
    try:
        if require_online is not None and not require_online():
            return {
                "sent": 0,
                "retries": 0,
                "latency_ms": [],
                "drain_ms": [],
                "limit": limit,
                "budget_ms": budget_ms,
            }
    except Exception:
        # If predicate misbehaves, proceed as if online
        pass
    db_path = DB_PATH
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    cur = conn.cursor()

    t0_ns = time.monotonic_ns()
    if now_ns is None:
        now_ns = t0_ns

    # fetch queued items
    cur.execute(
        "SELECT seq, ts_ns, payload FROM queue WHERE status=0 ORDER BY seq LIMIT ?",
        (limit,),
    )
    rows = cur.fetchall()

    # Measured latencies for this call
    e2e_lat_ms: list[float] = []
    drain_lat_ms: list[float] = []
    sent = 0
    retries = 0

    for seq, ts_ns, payload in rows:
        # budget check
        if (time.monotonic_ns() - t0_ns) / 1e6 >= budget_ms:
            break

        # measure drain-only latency around send_fn
        t_send_start_ns = time.monotonic_ns()
        try:
            send_fn(payload)  # append to out/sent.bin or raise if offline
        except Exception:
            # leave as queued; could implement retry/backoff counters here
            retries += 1
            continue
        t_send_end_ns = time.monotonic_ns()

        # mark sent
        cur.execute("UPDATE queue SET status=1 WHERE seq=?", (seq,))
        sent += 1

        # E2E: enqueue( ts_ns ) -> send_end
        e2e_ms = (now_ns - int(ts_ns)) / 1e6
        e2e_lat_ms.append(float(e2e_ms))

        # drain-only: send_start -> send_end
        drain_ms = (t_send_end_ns - t_send_start_ns) / 1e6
        drain_lat_ms.append(float(drain_ms))

    conn.commit()
    conn.close()

    return {
        "sent": sent,
        "retries": retries,
        "latency_ms": e2e_lat_ms,
        "drain_ms": drain_lat_ms,
        "limit": limit,
        "budget_ms": budget_ms,
    }
