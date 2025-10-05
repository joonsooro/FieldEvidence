from __future__ import annotations
import json, time
from pathlib import Path
from collections import deque
from typing import Deque, Optional

_METRICS_PATH = Path("out/metrics.json")


class RollingStats:
    """Rolling window for simple p50/p95."""
    def __init__(self, maxlen: int = 500):
        self.values: Deque[float] = deque(maxlen=maxlen)

    def add(self, v: float) -> None:
        self.values.append(float(v))

    def _pick(self, q: float) -> Optional[float]:
        if not self.values:
            return None
        arr = sorted(self.values)
        idx = int(q * (len(arr) - 1))
        return arr[idx]

    def p50(self) -> Optional[float]:
        return self._pick(0.50)

    def p95(self) -> Optional[float]:
        return self._pick(0.95)


class Metrics:
    """Collect stage latencies and persist p50/p95 breakdown to out/metrics.json."""
    def __init__(self, path: Path = _METRICS_PATH, window: int = 500):
        self.path = path
        # Keep legacy "decision_latency" for compatibility (maps to total)
        self._total = RollingStats(window)
        # New breakdown
        self._decode = RollingStats(window)
        self._estimate = RollingStats(window)
        self._risk = RollingStats(window)
        self._reco = RollingStats(window)

    # --- recorders ---
    def record_decode_ms(self, ms: float) -> None:
        self._decode.add(ms); self._flush()

    def record_estimate_ms(self, ms: float) -> None:
        self._estimate.add(ms); self._flush()

    def record_risk_ms(self, ms: float) -> None:
        self._risk.add(ms); self._flush()

    def record_reco_ms(self, ms: float) -> None:
        self._reco.add(ms); self._flush()

    def record_total_ms(self, ms: float) -> None:
        self._total.add(ms); self._flush()

    # Legacy alias (decision == total)
    def record_decision_latency_ms(self, ms: float) -> None:
        self.record_total_ms(ms)

    # --- persist ---
    def _flush(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "updated_ns": time.time_ns(),
            # legacy
            "decision_latency": {
                "p50_ms": self._total.p50(),
                "p95_ms": self._total.p95(),
                "n": len(self._total.values),
            },
            # new breakdown
            "breakdown": {
                "decode_ms":   {"p50": self._decode.p50(),   "p95": self._decode.p95()},
                "estimate_ms": {"p50": self._estimate.p50(), "p95": self._estimate.p95()},
                "risk_ms":     {"p50": self._risk.p50(),     "p95": self._risk.p95()},
                "reco_ms":     {"p50": self._reco.p50(),     "p95": self._reco.p95()},
                "total_ms":    {"p50": self._total.p50(),    "p95": self._total.p95()},
            },
        }
        self.path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def measure_decision(fn, *args, **kwargs):
    """Wrap callable and return (result, elapsed_ms)."""
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    ms = (time.perf_counter() - t0) * 1000.0
    return out, ms

