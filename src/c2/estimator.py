"""Lightweight 30s sliding-window tracker with time-decayed confidence.

This module provides a simple `Tracker` that maintains recent geo points and
estimates instantaneous velocity and heading from the two most recent points
in the window. Confidence decays over time since the last observation and
increases with the number of supporting points (up to three) and optional
per-observation confidence.

Formulas
--------
- Distance (meters): haversine formula on a sphere (R = 6_371_000 m).
- Bearing (degrees): initial navigation bearing from point 1 to point 2,
  normalized to [0, 360), where 0° = North and 90° = East.
- Confidence: exp(-Δt_last / tau) * min(1, N/3) * obs_conf, clamped to [0, 1].

Only Python stdlib is used.
"""

from __future__ import annotations

from collections import deque
from math import radians, sin, cos, asin, sqrt, atan2, degrees, exp
from typing import Deque, Dict, Optional, TypedDict


class _Point(TypedDict, total=False):
    ts_ns: int
    lat: float
    lon: float
    conf: float  # optional in inputs


def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two lat/lon points using haversine.

    Args:
        lat1, lon1: Latitude and longitude of the first point in degrees.
        lat2, lon2: Latitude and longitude of the second point in degrees.

    Returns:
        Distance in meters along the surface of a sphere of radius 6_371_000 m.
    """
    R = 6_371_000.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    lat1r = radians(lat1)
    lat2r = radians(lat2)

    a = sin(dlat / 2.0) ** 2 + cos(lat1r) * cos(lat2r) * sin(dlon / 2.0) ** 2
    c = 2.0 * asin(sqrt(a))
    return R * c


def _bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Initial navigation bearing from point 1 to point 2 in degrees.

    Bearing is normalized to [0, 360) with 0° = North and 90° = East.

    Args:
        lat1, lon1: Latitude and longitude of the first point in degrees.
        lat2, lon2: Latitude and longitude of the second point in degrees.

    Returns:
        Bearing in degrees in the range [0, 360).
    """
    lat1r = radians(lat1)
    lat2r = radians(lat2)
    dlon = radians(lon2 - lon1)

    y = sin(dlon) * cos(lat2r)
    x = cos(lat1r) * sin(lat2r) - sin(lat1r) * cos(lat2r) * cos(dlon)
    brng = (degrees(atan2(y, x)) + 360.0) % 360.0
    return brng


class Tracker:
    """Track recent points and estimate position/velocity with confidence.

    The tracker keeps points in a sliding time window relative to the most
    recent timestamp observed. Velocity is estimated from the last two points
    when available; otherwise it holds with zero speed and no heading.
    Confidence decays with time since the latest point and grows with the
    number of recent points (up to 3) and optional observation confidence.
    """

    def __init__(self, window_sec: float = 30.0, decay_tau_sec: float = 20.0) -> None:
        self.window_sec: float = float(window_sec)
        self.decay_tau_sec: float = float(decay_tau_sec)
        self._points: Deque[_Point] = deque()
        self._latest_ts_ns: Optional[int] = None

    def _prune(self) -> None:
        if self._latest_ts_ns is None:
            return
        window_ns = int(self.window_sec * 1e9)
        cutoff = self._latest_ts_ns - window_ns
        # Drop leftmost entries while older than cutoff.
        while self._points and self._points[0]["ts_ns"] < cutoff:
            self._points.popleft()

    def update(self, ev: Dict[str, object]) -> None:
        """Add a new observation and prune to the sliding window.

        Expected keys in `ev`:
            ts_ns (int), lat (float), lon (float)
        Optional:
            conf (float in [0, 1])
        """
        ts_ns_obj = ev.get("ts_ns")
        lat_obj = ev.get("lat")
        lon_obj = ev.get("lon")
        if not isinstance(ts_ns_obj, int) or not isinstance(lat_obj, (int, float)) or not isinstance(
            lon_obj, (int, float)
        ):
            # Silently ignore malformed updates to keep the tracker robust.
            return

        ts_ns = int(ts_ns_obj)
        lat = float(lat_obj)
        lon = float(lon_obj)

        point: _Point = {"ts_ns": ts_ns, "lat": lat, "lon": lon}
        if "conf" in ev:
            try:
                c = float(ev["conf"])  # type: ignore[index]
                # Clamp to [0, 1]
                if c < 0.0:
                    c = 0.0
                elif c > 1.0:
                    c = 1.0
                point["conf"] = c
            except (TypeError, ValueError):
                # Ignore invalid conf
                pass

        # Append point and update latest timestamp.
        self._points.append(point)
        self._latest_ts_ns = ts_ns if self._latest_ts_ns is None else max(self._latest_ts_ns, ts_ns)

        # Prune by window relative to latest timestamp seen so far.
        self._prune()

    def estimate(self, now_ns: int) -> Dict[str, object]:
        """Estimate current position, velocity, and confidence.

        Args:
            now_ns: Current time in nanoseconds for confidence decay.

        Returns a dict with keys:
            est_pos: {lat, lon}
            est_vel: {mps, heading_deg}
            confidence: float in [0, 1]
        """
        if not self._points:
            return {
                "est_pos": {"lat": 0.0, "lon": 0.0},
                "est_vel": {"mps": 0.0, "heading_deg": None},
                "confidence": 0.0,
            }

        last = self._points[-1]
        last_lat = float(last["lat"])
        last_lon = float(last["lon"])

        mps: float = 0.0
        heading: Optional[float] = None

        if len(self._points) >= 2:
            p1 = self._points[-2]
            p2 = self._points[-1]
            dt_ns = int(p2["ts_ns"]) - int(p1["ts_ns"])  # type: ignore[index]
            if dt_ns > 0:
                dt_s = dt_ns / 1e9
                dist_m = _haversine_m(float(p1["lat"]), float(p1["lon"]), float(p2["lat"]), float(p2["lon"]))
                if dist_m > 0.0:
                    mps = dist_m / dt_s
                    heading = _bearing_deg(float(p1["lat"]), float(p1["lon"]), float(p2["lat"]), float(p2["lon"]))
                else:
                    # No movement between points
                    mps = 0.0
                    heading = None
            else:
                # Non-positive dt: undefined, treat as stationary
                mps = 0.0
                heading = None

        # Confidence calculation
        latest_ts_ns = int(last["ts_ns"])  # type: ignore[index]
        dt_last_s = max(0.0, (now_ns - latest_ts_ns) / 1e9)
        time_decay = exp(-(dt_last_s / self.decay_tau_sec)) if self.decay_tau_sec > 0 else 0.0
        support = min(1.0, len(self._points) / 3.0)
        latest_conf = float(last.get("conf", 1.0))
        if latest_conf < 0.0:
            latest_conf = 0.0
        elif latest_conf > 1.0:
            latest_conf = 1.0
        confidence = time_decay * support * latest_conf
        if confidence < 0.0:
            confidence = 0.0
        elif confidence > 1.0:
            confidence = 1.0

        return {
            "est_pos": {"lat": last_lat, "lon": last_lon},
            "est_vel": {"mps": mps, "heading_deg": heading},
            "confidence": confidence,
        }


__all__ = ["Tracker", "_haversine_m", "_bearing_deg"]

