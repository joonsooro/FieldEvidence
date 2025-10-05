from __future__ import annotations
import json, time, statistics
from pathlib import Path
from collections import deque
from typing import Deque, Optional

_METRICS_PATH = Path("out/metrics.json")


class RollingStats:
    """Keeps a rolling window for simple p50/p95 calculations."""
    def __init__(self, maxlen: int = 500):
        self.values: Deque[float] = deque(maxlen=maxlen)

    def add(self, v: float) -> None:
        self.values.append(float(v))

    def p50(self) -> Optional[float]:
        if not self.values: return None
        arr = sorted(self.values)
        return arr[int(0.5 * (len(arr)-1))]

    def p95(self) -> Optional[float]:
        if not self.values: return None
        arr = sorted(self.values)
        return arr[int(0.95 * (len(arr)-1))]


class Metrics:
    """Collect decision latency and persist rolling p50/p95 to out/metrics.json."""
    def __init__(self, path: Path = _METRICS_PATH, window: int = 500):
        self.path = path
        self.decision_ms = RollingStats(window)

    def record_decision_latency_ms(self, ms: float) -> None:
        self.decision_ms.add(ms)
        self._flush()

    def _flush(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "updated_ns": time.time_ns(),
            "decision_latency": {
                "p50_ms": self.decision_ms.p50(),
                "p95_ms": self.decision_ms.p95(),
                "n": len(self.decision_ms.values)
            }
        }
        self.path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def measure_decision(fn, *args, **kwargs):
    """Wrap a callable and return (result, elapsed_ms)."""
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    ms = (time.perf_counter() - t0) * 1000.0
    return out, ms

